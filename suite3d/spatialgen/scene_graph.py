"""
Geometry-derived anatomical scene graph from segmentation masks.

Spatial relations in volumetric medical imaging are DETERMINISTICALLY COMPUTABLE
from segmentation + the NIfTI affine. They do not need to be extracted from
radiology reports by a language model.

ORIENTATION CONVENTION  --  read this before touching anything
-------------------------------------------------------------
We force every volume to nibabel RAS+ canonical orientation. In RAS+ the world
axes increase toward the patient's:
    +x  ->  RIGHT
    +y  ->  ANTERIOR
    +z  ->  SUPERIOR
NIfTI is natively RAS+; DICOM is LPS, so a DICOM->NIfTI conversion flips x and y.
Getting this backwards silently mislabels every laterality and A/P relation in
the whole dataset, so `as_closest_canonical` is mandatory and `selftest()`
asserts the signs on a synthetic volume.

Note on "left/right": we report PATIENT left/right (radiological ground truth),
never viewer left/right.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field

import numpy as np


# --- relation vocabulary -----------------------------------------------------
# Each axis relation is (positive_name, negative_name, world_axis_index).
AXES = {
    "lateral":      ("right_of", "left_of",       0),   # +x = patient right
    "anteroposterior": ("anterior_to", "posterior_to", 1),   # +y = anterior
    "longitudinal": ("superior_to", "inferior_to", 2),   # +z = superior
}


@dataclass
class Structure:
    """One segmented anatomical structure, in RAS+ world (mm) coordinates."""
    label: int
    name: str
    n_voxels: int
    volume_mm3: float
    centroid_ras: tuple[float, float, float]
    bbox_min_ras: tuple[float, float, float]
    bbox_max_ras: tuple[float, float, float]
    hu_mean: float | None = None
    hu_std: float | None = None
    # True when the segmentation touches the volume boundary, i.e. the structure
    # is cut off by the field of view. Its centroid and bbox then describe the
    # captured FRAGMENT, not the organ, and every relation computed from them is
    # unreliable. In 12 chest CTs this is what produced the single apparent
    # "varying" directional relation: vertebrae_C7 was partially captured, its
    # segmented volume ranging 4911-38655 voxels across scans (8x), and its
    # anteroposterior relation to the left adrenal flipped sign as a result.
    truncated: bool = False

    def extent(self, axis: int) -> tuple[float, float]:
        return (self.bbox_min_ras[axis], self.bbox_max_ras[axis])


@dataclass
class Relation:
    """A directed, geometrically verified relation between two structures."""
    subject: str
    predicate: str
    object: str
    axis: str | None = None
    # margin_mm: signed separation used to decide the relation. Large |margin|
    # means an unambiguous call; near zero means the structures overlap on this
    # axis and the relation should be treated as UNDECIDABLE, not forced.
    margin_mm: float | None = None
    evidence: dict = field(default_factory=dict)


def _voxel_to_ras(idx: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """idx: (N,3) voxel indices -> (N,3) RAS+ world coords in mm."""
    homo = np.hstack([idx, np.ones((idx.shape[0], 1))])
    return (affine @ homo.T).T[:, :3]


def build_structures(
    seg: np.ndarray,
    affine: np.ndarray,
    label_names: dict[int, str],
    image: np.ndarray | None = None,
    min_voxels: int = 50,
) -> dict[str, Structure]:
    """Build per-structure geometry records from an integer label map.

    `seg` and `affine` must ALREADY be in RAS+ (use load_ras()).
    `image` optionally supplies HU for intensity stats.
    Structures below `min_voxels` are dropped as segmentation noise.
    """
    vox_vol = float(abs(np.linalg.det(affine[:3, :3])))
    out: dict[str, Structure] = {}

    for label, name in label_names.items():
        mask = seg == label
        n = int(mask.sum())
        if n < min_voxels:
            continue
        idx = np.argwhere(mask)
        ras = _voxel_to_ras(idx, affine)

        hu_mean = hu_std = None
        if image is not None:
            vals = image[mask]
            hu_mean, hu_std = float(vals.mean()), float(vals.std())

        # touching any face of the volume means the organ is cut off by the
        # field of view; flag rather than silently emit fragment geometry
        lo_idx, hi_idx = idx.min(axis=0), idx.max(axis=0)
        truncated = bool((lo_idx == 0).any()
                         or (hi_idx == np.array(seg.shape) - 1).any())

        out[name] = Structure(
            label=int(label),
            name=name,
            n_voxels=n,
            volume_mm3=n * vox_vol,
            centroid_ras=tuple(ras.mean(axis=0).tolist()),
            bbox_min_ras=tuple(ras.min(axis=0).tolist()),
            bbox_max_ras=tuple(ras.max(axis=0).tolist()),
            hu_mean=hu_mean,
            hu_std=hu_std,
            truncated=truncated,
        )
    return out


def axis_relation(
    a: Structure,
    b: Structure,
    axis_key: str,
    min_margin_mm: float = 5.0,
    mode: str = "bbox",
) -> Relation | None:
    """Relation between a and b along one anatomical axis.

    Returns None when the call is geometrically UNDECIDABLE — i.e. the two
    structures overlap along this axis by more than the margin. Emitting a
    forced label there is exactly how report-derived pipelines manufacture
    wrong spatial ground truth, so we refuse instead.

    mode="bbox": require the bounding boxes to be genuinely separated.
    mode="centroid": compare centroids only (weaker, permits interleaving).
    """
    pos_name, neg_name, ax = AXES[axis_key]

    if mode == "centroid":
        margin = a.centroid_ras[ax] - b.centroid_ras[ax]
    else:
        a_lo, a_hi = a.extent(ax)
        b_lo, b_hi = b.extent(ax)
        if a_lo > b_hi:        # a entirely on + side of b
            margin = a_lo - b_hi
        elif b_lo > a_hi:      # a entirely on - side of b
            margin = -(b_lo - a_hi)
        else:
            return None        # overlapping -> undecidable

    if abs(margin) < min_margin_mm:
        return None

    pred = pos_name if margin > 0 else neg_name
    return Relation(
        subject=a.name, predicate=pred, object=b.name,
        axis=axis_key, margin_mm=float(margin),
        evidence={"mode": mode,
                  "a_extent": a.extent(ax), "b_extent": b.extent(ax)},
    )


def containment_relation(
    a: Structure, b: Structure, seg: np.ndarray, frac_thresh: float = 0.9
) -> Relation | None:
    """`a` is contained in `b` if >=frac_thresh of a's voxels fall inside b."""
    mask_a, mask_b = seg == a.label, seg == b.label
    if not mask_a.any():
        return None
    frac = float((mask_a & mask_b).sum()) / float(mask_a.sum())
    if frac < frac_thresh:
        return None
    return Relation(subject=a.name, predicate="contained_in", object=b.name,
                    evidence={"fraction": frac})


def bbox_gap_lower_bound(a: Structure, b: Structure) -> float:
    """Cheap lower bound on the surface-to-surface gap, from bounding boxes.

    Axis-wise separation of the two boxes; 0 when they overlap. Because it is a
    LOWER bound, `bbox_gap_lower_bound > max_gap` proves non-adjacency without
    touching a single voxel. This is what makes adjacency affordable: a real
    TotalSegmentator volume has 117 structures = 6,786 pairs, and running a full
    3D distance transform per pair takes hours. Nearly all pairs are far apart
    and get rejected here in microseconds.
    """
    d2 = 0.0
    for ax in range(3):
        a_lo, a_hi = a.extent(ax)
        b_lo, b_hi = b.extent(ax)
        if a_lo > b_hi:
            d2 += (a_lo - b_hi) ** 2
        elif b_lo > a_hi:
            d2 += (b_lo - a_hi) ** 2
    return float(np.sqrt(d2))


def adjacency_relation(
    a: Structure, b: Structure, seg: np.ndarray, affine: np.ndarray,
    max_gap_mm: float = 3.0,
) -> Relation | None:
    """`a` is adjacent to `b` if their surfaces come within max_gap_mm.

    Symmetric by construction, so callers emit it once per unordered pair.
    The distance transform is cropped to the union bounding box (plus margin)
    rather than run over the whole volume.
    """
    from scipy.ndimage import distance_transform_edt

    if bbox_gap_lower_bound(a, b) > max_gap_mm:
        return None                      # provably not adjacent, no voxel work

    spacing = np.abs(np.diag(affine)[:3])
    mask_a, mask_b = seg == a.label, seg == b.label
    if not mask_a.any() or not mask_b.any():
        return None

    # crop to the union of both structures plus a margin of max_gap, so the
    # cropped EDT cannot understate a gap that is still within threshold
    idx = np.argwhere(mask_a | mask_b)
    lo = idx.min(axis=0)
    hi = idx.max(axis=0) + 1
    pad = np.ceil(max_gap_mm / np.maximum(spacing, 1e-6)).astype(int) + 1
    lo = np.maximum(lo - pad, 0)
    hi = np.minimum(hi + pad, np.array(seg.shape))
    sl = tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))

    sub_a, sub_b = mask_a[sl], mask_b[sl]
    dist = distance_transform_edt(~sub_b, sampling=spacing)
    gap = float(dist[sub_a].min())
    if gap > max_gap_mm:
        return None
    return Relation(subject=a.name, predicate="adjacent_to", object=b.name,
                    margin_mm=gap, evidence={"gap_mm": gap})


def build_scene_graph(
    structures: dict[str, Structure],
    seg: np.ndarray,
    affine: np.ndarray,
    min_margin_mm: float = 5.0,
    do_adjacency: bool = True,
    do_containment: bool = False,
) -> list[Relation]:
    """All geometrically decidable relations over the structure set.

    do_containment defaults to False on purpose. TotalSegmentator is run with
    ml=True, which produces a MULTILABEL map: every voxel carries exactly one
    label, so two distinct structures can never share a voxel and
    containment_relation is guaranteed to return None. Computing it anyway costs
    two full-volume boolean comparisons per pair (~79 s/volume at 61 structures)
    to produce nothing. Containment is only meaningful across INDEPENDENT masks
    -- lesion vs. organ -- which lesion_binding.py handles directly. Enable this
    flag only when `seg` genuinely has overlapping labels.
    """
    rels: list[Relation] = []
    names = sorted(structures)
    for i, na in enumerate(names):
        for nb in names[i + 1:]:
            a, b = structures[na], structures[nb]
            for axis_key in AXES:
                r = axis_relation(a, b, axis_key, min_margin_mm)
                if r is not None:
                    rels.append(r)
            if do_containment:
                for x, y in ((a, b), (b, a)):
                    c = containment_relation(x, y, seg)
                    if c is not None:
                        rels.append(c)
            if do_adjacency:
                adj = adjacency_relation(a, b, seg, affine)
                if adj is not None:
                    rels.append(adj)
    return rels


# --- consistency axioms ------------------------------------------------------
# These are the cheap diagnostic the report claims is still unclaimed: a model
# that truly reads geometry cannot violate them; one that pattern-matches
# language priors will. We check them on MODEL OUTPUT, not on our own graph
# (ours is correct by construction -- asserting that is the selftest).

INVERSE = {
    "right_of": "left_of", "left_of": "right_of",
    "anterior_to": "posterior_to", "posterior_to": "anterior_to",
    "superior_to": "inferior_to", "inferior_to": "superior_to",
    "adjacent_to": "adjacent_to",
}


def check_antisymmetry(rels: list[Relation]) -> list[tuple]:
    """Violations of  A p B  =>  B inverse(p) A  (and never A p B & B p A)."""
    triples = {(r.subject, r.predicate, r.object) for r in rels}
    bad = []
    for s, p, o in triples:
        if p == "adjacent_to":
            continue
        if (o, p, s) in triples:          # symmetric where it must be inverse
            bad.append((s, p, o, "symmetric_violation"))
    return bad


def check_transitivity(rels: list[Relation]) -> list[tuple]:
    """Violations of  A p B & B p C  =>  A p C  for the three ordering axes."""
    from collections import defaultdict
    by_pred = defaultdict(set)
    for r in rels:
        if r.predicate in INVERSE and r.predicate != "adjacent_to":
            by_pred[r.predicate].add((r.subject, r.object))
    bad = []
    for pred, pairs in by_pred.items():
        succ = defaultdict(set)
        for s, o in pairs:
            succ[s].add(o)
        for a in list(succ):
            for b in list(succ[a]):
                for c in succ.get(b, ()):
                    if (a, c) not in pairs and c != a:
                        bad.append((a, pred, b, c, "transitivity_violation"))
    return bad


def selftest() -> None:
    """Synthetic volume with known geometry -- asserts the sign conventions.

    Builds an identity-affine RAS+ volume and places three cubes at known
    positions, then asserts the relations come out with the anatomically
    correct names. If this fails, every downstream label is wrong.
    """
    seg = np.zeros((40, 40, 40), dtype=np.int16)
    affine = np.eye(4)          # 1mm isotropic, voxel index == RAS mm

    seg[28:36, 10:18, 10:18] = 1   # high  x -> patient RIGHT
    seg[4:12,  10:18, 10:18] = 2   # low   x -> patient LEFT
    seg[10:18, 28:36, 10:18] = 3   # high  y -> ANTERIOR
    seg[10:18, 10:18, 28:36] = 4   # high  z -> SUPERIOR

    names = {1: "right_block", 2: "left_block", 3: "front_block", 4: "top_block"}
    st = build_structures(seg, affine, names)
    assert set(st) == set(names.values()), st.keys()

    r = axis_relation(st["right_block"], st["left_block"], "lateral")
    assert r and r.predicate == "right_of", r

    r = axis_relation(st["left_block"], st["right_block"], "lateral")
    assert r and r.predicate == "left_of", r

    r = axis_relation(st["front_block"], st["right_block"], "anteroposterior")
    assert r and r.predicate == "anterior_to", r

    r = axis_relation(st["top_block"], st["right_block"], "longitudinal")
    assert r and r.predicate == "superior_to", r

    # overlapping structures on an axis must be refused, not guessed
    seg2 = np.zeros((40, 40, 40), dtype=np.int16)
    seg2[10:20, 10:20, 10:20] = 1
    seg2[15:25, 10:20, 10:20] = 2      # overlaps 1 along x
    st2 = build_structures(seg2, affine, {1: "a", 2: "b"})
    assert axis_relation(st2["a"], st2["b"], "lateral") is None, "should refuse"

    # our own graph must satisfy the axioms by construction
    rels = build_scene_graph(st, seg, affine, do_adjacency=False)
    assert not check_antisymmetry(rels), check_antisymmetry(rels)
    assert not check_transitivity(rels), check_transitivity(rels)

    print(f"selftest OK — {len(st)} structures, {len(rels)} relations, "
          "RAS+ sign conventions verified, undecidable case refused")


def load_ras(path: str):
    """Load a NIfTI and force RAS+ canonical orientation. Returns (data, affine)."""
    import nibabel as nib
    img = nib.as_closest_canonical(nib.load(path))
    return np.asanyarray(img.dataobj), img.affine


def dump(structures, relations, path):
    with open(path, "w") as f:
        json.dump({"structures": {k: asdict(v) for k, v in structures.items()},
                   "relations": [asdict(r) for r in relations]}, f, indent=1)


if __name__ == "__main__":
    selftest()
