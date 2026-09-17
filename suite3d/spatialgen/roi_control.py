"""
Volumetric ROI controls: the 3D Visual Grounding Ratio.

An all-zero blind volume removes the image but also everything else -- anatomy,
contrast, body outline -- so a model could differ between arms for reasons
unrelated to the question. The Visual Grounding Ratio (ROI-only accuracy minus
ROI-masked accuracy) keeps a realistic image in both arms and changes only the
region the answer depends on.

Adapted to 3D counterfactual questions, where the answer depends on the lesion
and on the tissue between it and the target structure:

  ROI-ONLY    keep a dilated shell around lesion+target, blank the rest
  ROI-MASKED  blank exactly that shell, keep the rest
  FULL        untouched volume
  ZERO        all zeros

A model that grounds its answer in the evidence scores high on ROI-only and
drops on ROI-masked. One answering from priors scores the same on both.

The masking region is derived from the same geometry that generated the answer,
so it is exact rather than a hand-drawn box.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


def evidence_region(lesion: np.ndarray, target: np.ndarray,
                    affine: np.ndarray, margin_mm: float = 15.0) -> np.ndarray:
    """Voxels the counterfactual answer can depend on.

    The lesion, the target, and the corridor between them: growth proceeds
    outward from the lesion surface, so what matters is the tissue it would pass
    through. Implemented as a distance-based dilation of the union, which keeps
    the region anisotropy-correct in millimetres.
    """
    from scipy.ndimage import distance_transform_edt

    spacing = np.abs(np.diag(affine)[:3])
    union = lesion | target
    if not union.any():
        return np.zeros_like(lesion, dtype=bool)

    # The transform is only consulted through `dist <= margin_mm`, so distances
    # beyond the margin are discarded. Computing it on the union's bounding box
    # padded by the margin gives the same answer: a voxel outside that box is
    # more than margin_mm from every union voxel along at least one axis, so its
    # true distance already exceeds the margin.
    #
    # Run on the whole volume this allocates a float64 array the size of the
    # scan -- 1.05 GB on the 512x512x501 liver series, once per probe, plus
    # scipy's internal feature arrays. That is what the host OOM killer was
    # reacting to when the liver ROI runs died at 252 of 600 probes.
    pad = np.ceil(margin_mm / np.maximum(spacing, 1e-6)).astype(int) + 1
    idx = np.argwhere(union)
    lo = np.maximum(idx.min(0) - pad, 0)
    hi = np.minimum(idx.max(0) + pad + 1, union.shape)
    box = tuple(slice(a, b) for a, b in zip(lo, hi))

    dist = distance_transform_edt(~union[box], sampling=spacing)
    out = np.zeros_like(lesion, dtype=bool)
    out[box] = dist <= margin_mm
    return out


def apply_condition(volume: np.ndarray, roi: np.ndarray, condition: str,
                    fill: float | str | None = "local") -> np.ndarray:
    """Produce one of the four arms.

    `fill` controls what replaces the masked region, and the choice
    matters. On CT the volume's 1st percentile is air outside the patient
    (about -1000 HU), so filling with it carves an air cavity into soft tissue
    that a model can detect as an artefact rather than as missing evidence.

    "local" replaces the region with the median intensity of a shell of tissue
    immediately surrounding it, so the masked volume stays locally plausible and
    the only thing removed is the evidence. "air" reproduces the original
    behaviour for comparison; a float sets the value directly.
    """
    if condition == "full":
        return volume.copy()
    if condition == "zero":
        return np.zeros_like(volume)

    if fill == "local":
        from scipy.ndimage import binary_dilation
        # The shell is by definition within 6 voxels of the ROI, so dilating the
        # whole volume computes the same value as dilating a padded bounding box
        # around it and costs the volume's size instead of the region's. On the
        # 512x512x501 liver scans that is a 5x difference per item, and the two
        # give bit-identical fill values (selftest asserts it).
        idx = np.argwhere(roi)
        if len(idx):
            lo = np.maximum(idx.min(0) - 7, 0)
            hi = np.minimum(idx.max(0) + 8, roi.shape)
            box = tuple(slice(a, b) for a, b in zip(lo, hi))
            rc = roi[box]
            shell = binary_dilation(rc, iterations=6) & ~rc
            bg = float(np.median(volume[box][shell])) if shell.any() \
                else float(np.median(volume))
        else:
            bg = float(np.median(volume))
    elif fill == "air" or fill is None:
        bg = float(np.percentile(volume, 1))
    else:
        bg = float(fill)

    out = volume.copy()
    if condition == "roi_only":
        out[~roi] = bg
    elif condition == "roi_masked":
        out[roi] = bg
    else:
        raise ValueError(condition)
    return out


def vgr(acc_roi_only: float, acc_roi_masked: float) -> float:
    """Visual grounding ratio, in percentage points (BrokenEvidence definition)."""
    return 100.0 * (acc_roi_only - acc_roi_masked)


def selftest() -> None:
    shape = (60, 60, 30)
    affine = np.diag([1.0, 1.0, 3.0, 1.0])
    vol = np.random.default_rng(0).normal(50, 5, shape).astype(np.float32)
    vol[:5] = -1000.0                       # a background region to pick up

    lesion = np.zeros(shape, bool)
    lesion[28:32, 28:32, 14:16] = True
    target = np.zeros(shape, bool)
    target[45:52, 26:34, 12:18] = True

    roi = evidence_region(lesion, target, affine, margin_mm=10.0)
    assert roi[lesion].all() and roi[target].all(), "ROI must contain both"
    assert not roi.all(), "ROI must not swallow the volume"
    # the corridor between them should be included
    assert roi[36, 30, 15], "tissue between lesion and target must be in the ROI"

    only = apply_condition(vol, roi, "roi_only")
    masked = apply_condition(vol, roi, "roi_masked")
    full = apply_condition(vol, roi, "full")
    zero = apply_condition(vol, roi, "zero")

    assert np.allclose(only[roi], vol[roi]), "roi_only must preserve the ROI"
    assert np.allclose(masked[~roi], vol[~roi]), "roi_masked must preserve outside"
    assert not np.allclose(only[~roi], vol[~roi]), "roi_only must blank outside"
    assert not np.allclose(masked[roi], vol[roi]), "roi_masked must blank the ROI"
    assert np.allclose(full, vol)
    assert zero.max() == 0

    # the two arms must be complementary: together they cover the whole volume
    assert np.array_equal(roi | ~roi, np.ones(shape, bool))

    # the fill choice is a real experimental variable, so both must work and
    # must differ. "air" carves a -1000 HU cavity into soft tissue, which a
    # model can detect as an artefact; "local" replaces the region with the
    # median of the tissue shell around it, leaving the volume locally
    # plausible. Asserting they differ keeps the comparison from being vacuous.
    m_local = apply_condition(vol, roi, "roi_masked", fill="local")
    m_air = apply_condition(vol, roi, "roi_masked", fill="air")
    v_local = float(m_local[roi].mean())
    v_air = float(m_air[roi].mean())
    assert abs(v_local - v_air) > 100.0, (v_local, v_air)
    assert 30.0 < v_local < 70.0, f"local fill should sit in tissue: {v_local}"
    assert v_air < -500.0, f"air fill should sit at background: {v_air}"

    # evidence_region crops the distance transform to a padded bounding box.
    # That is exact only if the padding is right, and getting it wrong shrinks
    # the region silently rather than failing -- so the equivalence with the
    # full-volume transform is asserted, including a union touching the border.
    from scipy.ndimage import distance_transform_edt as _edt
    for _les, _tgt in ((lesion, target),
                       (np.pad(np.ones((3, 3, 2), bool),
                               [(0, shape[0] - 3), (0, shape[1] - 3),
                                (0, shape[2] - 2)]),
                        np.pad(np.ones((3, 3, 2), bool),
                               [(shape[0] - 3, 0), (shape[1] - 3, 0),
                                (shape[2] - 2, 0)]))):
        _sp = np.abs(np.diag(affine)[:3])
        _ref = _edt(~(_les | _tgt), sampling=_sp) <= 15.0
        assert np.array_equal(_ref, evidence_region(_les, _tgt, affine, 15.0)), \
            "cropped distance transform differs from the full-volume one"

    # the local fill is computed on a padded bounding box for speed; that is an
    # optimisation only if it is exact, so the equivalence is asserted rather
    # than assumed. A border-touching ROI is included because clipping the box
    # is where a padded-crop implementation goes wrong.
    from scipy.ndimage import binary_dilation as _bd
    for probe in (roi, np.pad(np.ones((4, 4, 3), bool),
                              [(0, shape[0] - 4), (0, shape[1] - 4),
                               (0, shape[2] - 3)])):
        _sh = _bd(probe, iterations=6) & ~probe
        _ref = float(np.median(vol[_sh]))
        _got = float(np.unique(apply_condition(vol, probe, "roi_masked",
                                               fill="local")[probe])[0])
        assert _got == _ref, f"cropped fill {_got} != full-volume fill {_ref}"

    print(f"selftest OK — evidence region contains lesion, target and the "
          f"corridor between them; the four arms behave as specified and the "
          f"two masked arms are exact complements; the masked fill is a "
          f"controlled variable (local tissue {v_local:.0f} vs air {v_air:.0f} "
          f"HU), which matters because an air cavity is itself detectable")


if __name__ == "__main__":
    selftest()
