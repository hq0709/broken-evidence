"""
Finding-to-anatomy binding with computed ground truth.

Clinical spatial questions rarely relate two normal organs; they BIND A FINDING
TO AN ANATOMICAL FRAME ("where is the nodule", "does the tumour reach the
aorta"). That task is fully verifiable IF there is a lesion mask, which the
Medical Segmentation Decathlon tasks provide (Lung/Liver/Pancreas/Colon
tumours). Pairing an MSD lesion mask with a TotalSegmentator anatomy
segmentation yields finding->anatomy QA with computed ground truth, at zero
labelling cost.

Label conventions differ per MSD task, hence LESION_LABEL below.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field

import numpy as np

from scene_graph import Structure, _voxel_to_ras

# MSD tasks: which integer label in labelsTr marks the lesion.
LESION_LABEL = {
    "Task03_Liver": 2,      # 1 = liver, 2 = tumour
    "Task06_Lung": 1,       # 1 = cancer
    "Task07_Pancreas": 2,   # 1 = pancreas, 2 = tumour
    "Task10_Colon": 1,      # 1 = cancer
}

# Surface gap at or below which a "contact" claim is defensible. Kept tight on
# purpose: it is the wording of the generated question that has to be true, and
# CT voxel spacing alone is often ~1 mm, so anything larger is proximity, not
# contact.
CONTACT_MM = 1.5

# Structures that can plausibly CONTAIN a lesion. Restricting to these avoids
# reporting that a nodule is "inside" a rib merely because bboxes overlap.
CONTAINER_WHITELIST = {
    "lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
    "lung_middle_lobe_right", "lung_lower_lobe_right",
    "liver", "spleen", "pancreas", "kidney_left", "kidney_right",
    "stomach", "colon", "small_bowel", "esophagus", "urinary_bladder",
    "adrenal_gland_left", "adrenal_gland_right", "gallbladder", "prostate",
    "thyroid_gland", "brain", "heart", "spinal_cord",
}


@dataclass
class Lesion:
    lesion_id: str
    n_voxels: int
    volume_mm3: float
    centroid_ras: tuple[float, float, float]
    bbox_min_ras: tuple[float, float, float]
    bbox_max_ras: tuple[float, float, float]
    max_diameter_mm: float
    # binding results
    container: str | None = None          # organ with largest voxel overlap
    container_frac: float = 0.0           # fraction of lesion inside container
    laterality: str | None = None         # left / right / midline
    neighbours: list = field(default_factory=list)   # (name, gap_mm)
    ambiguous: bool = False               # True when binding is not decidable


def find_lesions(
    lesion_mask: np.ndarray, affine: np.ndarray, min_voxels: int = 20
) -> list[tuple[str, np.ndarray]]:
    """Split a binary lesion mask into connected components.

    Tiny components are dropped: below ~20 voxels they are usually annotation
    speckle, and generating QA about them manufactures unanswerable questions.
    """
    from scipy.ndimage import label as cc_label

    lab, n = cc_label(lesion_mask)
    out = []
    for i in range(1, n + 1):
        m = lab == i
        if int(m.sum()) >= min_voxels:
            out.append((f"lesion{i}", m))
    return out


def _max_diameter(ras: np.ndarray, cap: int = 4000) -> float:
    """Largest pairwise distance across the lesion surface, in mm.

    Subsamples above `cap` points -- the exact diameter needs an O(n^2) pass and
    lesions can exceed 10^5 voxels; a 4k subsample is well within annotation
    noise for a quantity radiologists report to the nearest millimetre.
    """
    if len(ras) > cap:
        idx = np.random.default_rng(0).choice(len(ras), cap, replace=False)
        ras = ras[idx]
    if len(ras) < 2:
        return 0.0
    from scipy.spatial.distance import pdist
    return float(pdist(ras).max())


def bind_lesion(
    lesion_id: str,
    mask: np.ndarray,
    anatomy_seg: np.ndarray,
    affine: np.ndarray,
    label_names: dict[int, str],
    structures: dict[str, Structure],
    midline_x: float,
    min_container_frac: float = 0.5,
    neighbour_gap_mm: float = 5.0,
) -> Lesion:
    """Compute which anatomical structure a lesion sits in, plus its neighbours."""
    from scipy.ndimage import distance_transform_edt

    idx = np.argwhere(mask)
    ras = _voxel_to_ras(idx, affine)
    vox_vol = float(abs(np.linalg.det(affine[:3, :3])))
    n = int(mask.sum())

    les = Lesion(
        lesion_id=lesion_id, n_voxels=n, volume_mm3=n * vox_vol,
        centroid_ras=tuple(ras.mean(axis=0).tolist()),
        bbox_min_ras=tuple(ras.min(axis=0).tolist()),
        bbox_max_ras=tuple(ras.max(axis=0).tolist()),
        max_diameter_mm=_max_diameter(ras),
    )

    # --- container: structure holding the largest share of lesion voxels ------
    overlaps: list[tuple[str, float]] = []
    inside = anatomy_seg[mask]
    for lab, cnt in zip(*np.unique(inside, return_counts=True)):
        name = label_names.get(int(lab))
        if name and name in CONTAINER_WHITELIST:
            overlaps.append((name, float(cnt) / n))
    overlaps.sort(key=lambda t: -t[1])

    if overlaps and overlaps[0][1] >= min_container_frac:
        les.container, les.container_frac = overlaps[0]
        # a lesion split near-evenly across two structures is genuinely
        # ambiguous (e.g. straddling a fissure); flag rather than pick one
        if len(overlaps) > 1 and overlaps[1][1] > 0.35:
            les.ambiguous = True
    elif overlaps:
        les.container, les.container_frac = overlaps[0]
        les.ambiguous = True
    else:
        les.ambiguous = True

    # --- laterality relative to body midline ---------------------------------
    lo, hi = les.bbox_min_ras[0], les.bbox_max_ras[0]
    if lo > midline_x:
        les.laterality = "right"
    elif hi < midline_x:
        les.laterality = "left"
    else:
        les.laterality = "midline"

    # --- nearby structures ----------------------------------------------------
    # Same cost trap as scene_graph.adjacency_relation: a full-volume distance
    # transform per structure is unaffordable at 117 structures. Reject on
    # bounding boxes first, then run the EDT on a crop.
    spacing = np.abs(np.diag(affine)[:3])
    les_lo = np.array(les.bbox_min_ras)
    les_hi = np.array(les.bbox_max_ras)

    for name, s in structures.items():
        if name == les.container:
            continue
        s_lo, s_hi = np.array(s.bbox_min_ras), np.array(s.bbox_max_ras)
        # axis-wise bbox separation is a lower bound on the true surface gap
        sep = np.maximum(np.maximum(s_lo - les_hi, les_lo - s_hi), 0.0)
        if float(np.linalg.norm(sep)) > neighbour_gap_mm:
            continue

        om = anatomy_seg == s.label
        if not om.any():
            continue
        idx_u = np.argwhere(om | mask)
        lo = idx_u.min(axis=0)
        hi = idx_u.max(axis=0) + 1
        pad = np.ceil(neighbour_gap_mm / np.maximum(spacing, 1e-6)).astype(int) + 1
        lo = np.maximum(lo - pad, 0)
        hi = np.minimum(hi + pad, np.array(anatomy_seg.shape))
        sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))

        d = distance_transform_edt(~om[sl], sampling=spacing)
        gap = float(d[mask[sl]].min())
        if gap <= neighbour_gap_mm:
            les.neighbours.append([name, round(gap, 2)])
    les.neighbours.sort(key=lambda t: t[1])
    return les


def lesion_qa(les: Lesion, volume_id: str) -> list[dict]:
    """Verified QA about one lesion. Ambiguous bindings produce no QA at all."""
    out = []
    base = f"{volume_id}_{les.lesion_id}"

    if les.container and not les.ambiguous:
        out.append({
            "qid": f"{base}_container", "category": "adjacency",
            "question": (f"In which anatomical structure is the lesion "
                         f"located?"),
            "answer": les.container.replace("_", " "),
            "provenance": {"rule": "max voxel overlap",
                           "fraction": round(les.container_frac, 3)},
        })

    if les.laterality in ("left", "right"):
        out.append({
            "qid": f"{base}_lat", "category": "laterality",
            "question": ("Is the lesion on the patient's left or right side? "
                         "Answer with one word."),
            "answer": les.laterality, "choices": ["left", "right"],
            "provenance": {"rule": "midline comparison",
                           "x_extent": [les.bbox_min_ras[0], les.bbox_max_ras[0]]},
        })

    out.append({
        "qid": f"{base}_size", "category": "extent",
        "question": "What is the approximate maximum diameter of the lesion in millimetres?",
        "answer": f"{les.max_diameter_mm:.1f}",
        "provenance": {"rule": "max pairwise surface distance",
                       "volume_mm3": round(les.volume_mm3, 1)},
    })

    # Wording must match the measured threshold. Asking "does it contact X?" and
    # answering yes off a 4.85 mm gap manufactures a wrong label: no radiologist
    # calls a nodule 5 mm from a rib "contacting" it. So a contact claim is made
    # only within CONTACT_MM, and anything looser becomes an explicitly
    # quantified proximity question, which is unambiguous at any distance.
    for name, gap in les.neighbours[:3]:
        pretty = name.replace("_", " ")
        if gap <= CONTACT_MM:
            out.append({
                "qid": f"{base}_contact_{name}", "category": "adjacency",
                "question": (f"Does the lesion contact the {pretty}? "
                             f"Answer yes or no."),
                "answer": "yes", "choices": ["no", "yes"],
                "provenance": {"rule": "surface distance",
                               "gap_mm": gap, "threshold_mm": CONTACT_MM},
            })
        else:
            bound = int(np.ceil(gap / 5.0) * 5)   # round up to a clean 5 mm step
            out.append({
                "qid": f"{base}_near_{name}", "category": "adjacency",
                "question": (f"Is the lesion within {bound} mm of the {pretty}? "
                             f"Answer yes or no."),
                "answer": "yes", "choices": ["no", "yes"],
                "provenance": {"rule": "surface distance",
                               "gap_mm": gap, "threshold_mm": bound},
            })
    return out


def selftest() -> None:
    """Synthetic lung-like volume: two lobes, one lesion inside the right lobe."""
    from scene_graph import build_structures

    shape = (200, 100, 100)
    anat = np.zeros(shape, dtype=np.int16)
    anat[110:190, 20:80, 20:80] = 1        # right lobe (high x = patient right)
    anat[10:90, 20:80, 20:80] = 2          # left lobe
    names = {1: "lung_upper_lobe_right", 2: "lung_upper_lobe_left"}
    affine = np.eye(4)

    lesion = np.zeros(shape, dtype=bool)
    lesion[140:150, 40:50, 40:50] = True   # fully inside the right lobe

    st = build_structures(anat, affine, names)
    midline = (st["lung_upper_lobe_right"].centroid_ras[0]
               + st["lung_upper_lobe_left"].centroid_ras[0]) / 2

    comps = find_lesions(lesion, affine)
    assert len(comps) == 1, comps
    les = bind_lesion(comps[0][0], comps[0][1], anat, affine, names, st, midline)

    assert les.container == "lung_upper_lobe_right", les.container
    assert les.container_frac == 1.0, les.container_frac
    assert not les.ambiguous
    assert les.laterality == "right", les.laterality
    # 10x10x10 voxel cube -> body diagonal of a 9mm cube ~ 15.6mm
    assert 14 < les.max_diameter_mm < 17, les.max_diameter_mm

    qas = lesion_qa(les, "synth")
    cats = {q["category"] for q in qas}
    assert "laterality" in cats and "extent" in cats, cats

    # a lesion straddling both lobes must be flagged ambiguous, not forced
    straddle = np.zeros(shape, dtype=bool)
    straddle[85:115, 40:50, 40:50] = True
    c2 = find_lesions(straddle, affine)
    l2 = bind_lesion(c2[0][0], c2[0][1], anat, affine, names, st, midline)
    assert l2.ambiguous, "straddling lesion should be ambiguous"
    assert not any(q["qid"].endswith("_container") for q in lesion_qa(l2, "synth")), \
        "ambiguous lesion must not emit a container QA"

    print(f"selftest OK — bound lesion to {les.container} "
          f"(frac {les.container_frac:.2f}, {les.laterality}, "
          f"{les.max_diameter_mm:.1f}mm), {len(qas)} QA; ambiguous case refused")


if __name__ == "__main__":
    selftest()
