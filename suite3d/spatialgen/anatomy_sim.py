"""
A deterministic anatomical simulator, exposed as agent tools.

The gap this fills
------------------
Every medical-imaging agent published so far calls tools that MEASURE THE
PRESENT STATE: segment, measure, crop, look up. None can answer a question about
a state that does not exist -- "if this lesion grew 20%, would it reach the
aorta?", "if we resect the right lower lobe, how much lung remains?". Those are
the questions surgical and oncological planning actually turn on.

Why counterfactual queries are the right target here
----------------------------------------------------
Two failure modes sank the more obvious ideas in this project, and simulation is
immune to both:

  * Directional relations between normal organs are constant across patients
    (measured: 3651 relations over abdominal and chest CT, zero varying), so
    questions about them carry no patient-specific information. A counterfactual
    is patient-specific by construction -- it depends on this lesion's shape and
    position.
  * Benchmarks built from radiology reports inherit whatever the report says and
    cannot be checked. Here the simulator IS the ground truth: the answer is
    recomputed from masks, so it is verifiable to the voxel.

And no report, textbook, or language prior can answer a counterfactual, because
the scenario never occurred. A blind model has nothing to fall back on.

Everything below is deterministic mask arithmetic -- no training, no generation.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


@dataclass
class SimResult:
    query: str
    answer: str
    evidence: dict


def _spacing(affine: np.ndarray) -> np.ndarray:
    return np.abs(np.diag(affine)[:3])


def grow(mask: np.ndarray, affine: np.ndarray, mm: float) -> np.ndarray:
    """Isotropic growth of a lesion by `mm` millimetres.

    Uses a distance transform rather than binary dilation: dilation works in
    voxels, and CT voxels are routinely anisotropic (0.7x0.7x5 mm here), so a
    voxel-based dilation would grow ~7x further through-plane than in-plane and
    silently produce a physically wrong shape.
    """
    from scipy.ndimage import distance_transform_edt

    if mm <= 0:
        return mask.copy()
    dist = distance_transform_edt(~mask, sampling=_spacing(affine))
    return dist <= mm


def shrink(mask: np.ndarray, affine: np.ndarray, mm: float) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt

    if mm <= 0:
        return mask.copy()
    inner = distance_transform_edt(mask, sampling=_spacing(affine))
    return inner > mm


def volume_mm3(mask: np.ndarray, affine: np.ndarray) -> float:
    return float(mask.sum() * abs(np.linalg.det(affine[:3, :3])))


def surface_gap_mm(a: np.ndarray, b: np.ndarray, affine: np.ndarray) -> float:
    """Minimum surface-to-surface distance; 0.0 when they overlap."""
    from scipy.ndimage import distance_transform_edt

    if not a.any() or not b.any():
        return float("inf")
    if (a & b).any():
        return 0.0
    dist = distance_transform_edt(~b, sampling=_spacing(affine))
    return float(dist[a].min())


# --- the agent-facing tools ---------------------------------------------------

def tool_growth_contact(lesion: np.ndarray, target: np.ndarray,
                        affine: np.ndarray, growth_mm: float,
                        target_name: str = "target") -> SimResult:
    """Would the lesion contact `target` after growing by `growth_mm`?"""
    before = surface_gap_mm(lesion, target, affine)
    grown = grow(lesion, affine, growth_mm)
    after = surface_gap_mm(grown, target, affine)
    contacts = after <= 0.0
    return SimResult(
        query=(f"If the lesion grew by {growth_mm:g} mm in every direction, "
               f"would it contact the {target_name.replace('_', ' ')}?"),
        answer="yes" if contacts else "no",
        evidence={"gap_before_mm": round(before, 2),
                  "gap_after_mm": round(after, 2),
                  "growth_mm": growth_mm,
                  "rule": "surface distance after isotropic growth"},
    )


def tool_growth_to_contact(lesion: np.ndarray, target: np.ndarray,
                           affine: np.ndarray, target_name: str = "target",
                           max_mm: float = 60.0, step: float = 1.0) -> SimResult:
    """How much growth until contact? Answers the inverse question."""
    gap = surface_gap_mm(lesion, target, affine)
    needed = 0.0 if gap <= 0 else min(gap, max_mm)
    return SimResult(
        query=(f"By how many millimetres would the lesion have to grow before "
               f"it contacted the {target_name.replace('_', ' ')}?"),
        answer=f"{needed:.1f}",
        evidence={"current_gap_mm": round(gap, 2), "capped_at_mm": max_mm,
                  "rule": "growth needed == current surface gap"},
    )


def tool_resection_remaining(organ_parts: dict[str, np.ndarray],
                             affine: np.ndarray, resected: str) -> SimResult:
    """What fraction of the organ remains after removing one part?"""
    total = sum(volume_mm3(m, affine) for m in organ_parts.values())
    removed = volume_mm3(organ_parts[resected], affine)
    frac = (total - removed) / total if total else 0.0
    return SimResult(
        query=(f"If the {resected.replace('_', ' ')} were resected, what "
               f"percentage of the organ volume would remain?"),
        answer=f"{100 * frac:.1f}",
        evidence={"total_mm3": round(total, 1), "removed_mm3": round(removed, 1),
                  "parts": sorted(organ_parts),
                  "rule": "volume ratio after removing one part"},
    )


def tool_displacement_contact(lesion: np.ndarray, target: np.ndarray,
                              affine: np.ndarray, shift_mm: tuple,
                              target_name: str = "target") -> SimResult:
    """Mass effect: shift the lesion and re-test contact.

    Shifts are given in millimetres along RAS axes and converted to voxels with
    the real spacing, so an anisotropic volume does not distort the motion.
    """
    from scipy.ndimage import shift as nd_shift

    sp = _spacing(affine)
    vox = [s / p for s, p in zip(shift_mm, sp)]
    moved = nd_shift(lesion.astype(np.float32), vox, order=0) > 0.5
    before = surface_gap_mm(lesion, target, affine)
    after = surface_gap_mm(moved, target, affine)
    return SimResult(
        query=(f"If the lesion were displaced by {shift_mm} mm (RAS), would it "
               f"contact the {target_name.replace('_', ' ')}?"),
        answer="yes" if after <= 0.0 else "no",
        evidence={"gap_before_mm": round(before, 2),
                  "gap_after_mm": round(after, 2), "shift_mm": list(shift_mm),
                  "rule": "surface distance after rigid displacement"},
    )


def selftest() -> None:
    """Anisotropic phantom: growth and displacement must respect real spacing."""
    shape = (60, 60, 30)
    affine = np.diag([1.0, 1.0, 4.0, 1.0])     # 4 mm slices: strongly anisotropic

    lesion = np.zeros(shape, dtype=bool)
    lesion[28:32, 28:32, 14:16] = True
    target = np.zeros(shape, dtype=bool)
    target[45:55, 25:35, 10:20] = True          # ~14 mm away in +x

    gap = surface_gap_mm(lesion, target, affine)
    assert 10 < gap < 20, gap

    # growing less than the gap must NOT contact; more than it must
    r_small = tool_growth_contact(lesion, target, affine, 5.0, "aorta")
    r_big = tool_growth_contact(lesion, target, affine, 25.0, "aorta")
    assert r_small.answer == "no", r_small
    assert r_big.answer == "yes", r_big
    assert r_small.evidence["gap_after_mm"] > 0

    # the inverse query must return the current gap
    r_inv = tool_growth_to_contact(lesion, target, affine, "aorta")
    assert abs(float(r_inv.answer) - gap) < 1.5, (r_inv.answer, gap)

    # growth must be isotropic in MILLIMETRES, not voxels: with 4 mm slices a
    # voxel-based dilation would reach ~4x further in z. Check the grown extent
    # is comparable along x and z in physical units.
    grown = grow(lesion, affine, 8.0)
    idx = np.argwhere(grown)
    ext_x = (idx[:, 0].max() - idx[:, 0].min()) * 1.0
    ext_z = (idx[:, 2].max() - idx[:, 2].min()) * 4.0
    assert abs(ext_x - ext_z) < 6.0, (ext_x, ext_z, "growth is not isotropic in mm")

    # resection arithmetic
    parts = {"lobe_a": np.zeros(shape, bool), "lobe_b": np.zeros(shape, bool)}
    parts["lobe_a"][0:10, :, :] = True
    parts["lobe_b"][10:40, :, :] = True
    r_res = tool_resection_remaining(parts, affine, "lobe_a")
    assert abs(float(r_res.answer) - 75.0) < 0.5, r_res.answer

    # displacement toward the target closes the gap
    r_far = tool_displacement_contact(lesion, target, affine, (14, 0, 0), "aorta")
    assert r_far.answer == "yes", r_far

    print(f"selftest OK — gap {gap:.1f}mm; growth 5mm->no / 25mm->yes; "
          f"inverse query returns {float(r_inv.answer):.1f}mm; growth isotropic "
          f"in mm on a 4mm-slice volume (x {ext_x:.0f} vs z {ext_z:.0f}); "
          f"resection leaves {r_res.answer}%")


if __name__ == "__main__":
    selftest()
