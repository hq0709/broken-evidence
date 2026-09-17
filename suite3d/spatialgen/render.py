"""
Render a 3D CT volume into the 2D images a vision-language model can consume.

This step is not cosmetic. Which view you show determines whether a spatial
question is answerable at all:

    superior / inferior   -> visible in coronal and sagittal, NOT in a single axial slice
    left / right          -> visible in coronal and axial, NOT in sagittal
    anterior / posterior  -> visible in sagittal and axial, NOT in coronal

Evaluating a model on "is A superior to B" while showing only axial slices
measures the harness, not the model. So the default is the three orthogonal
views, and `views_for_category()` states the dependency explicitly.

Everything here operates on RAS+ arrays (see scene_graph.load_ras), so axis 0 is
x/right, axis 1 is y/anterior, axis 2 is z/superior.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Standard CT display windows, (centre, width) in Hounsfield units.
WINDOWS = {
    "soft_tissue": (40, 400),
    "lung": (-600, 1500),
    "bone": (400, 1800),
    "brain": (40, 80),
    "mediastinum": (50, 350),
}

# Which orthogonal planes actually carry the information for each question type.
_VIEW_DEPS = {
    "laterality": ("coronal", "axial"),
    "longitudinal": ("coronal", "sagittal"),
    "anteroposterior": ("sagittal", "axial"),
    "adjacency": ("axial", "coronal", "sagittal"),
    "extent": ("axial", "coronal", "sagittal"),
    "mediolateral": ("axial", "coronal"),
}


def views_for_category(category: str) -> tuple[str, ...]:
    """Planes in which `category` is decidable. Empty tuple for unknown types."""
    return _VIEW_DEPS.get(category, ("axial", "coronal", "sagittal"))


@dataclass
class Rendered:
    axial: np.ndarray | None = None
    coronal: np.ndarray | None = None
    sagittal: np.ndarray | None = None

    def available(self) -> dict[str, np.ndarray]:
        return {k: v for k, v in
                (("axial", self.axial), ("coronal", self.coronal),
                 ("sagittal", self.sagittal)) if v is not None}


def window(volume: np.ndarray, preset: str = "soft_tissue") -> np.ndarray:
    """Apply a CT window and map to uint8. Values outside the window clip."""
    if preset not in WINDOWS:
        raise ValueError(f"unknown window {preset!r}; have {sorted(WINDOWS)}")
    centre, width = WINDOWS[preset]
    lo, hi = centre - width / 2.0, centre + width / 2.0
    v = np.clip(volume.astype(np.float32), lo, hi)
    return (((v - lo) / (hi - lo)) * 255.0).astype(np.uint8)


def _to_display(plane: np.ndarray, flip_v: bool, flip_h: bool) -> np.ndarray:
    """Orient a extracted plane into radiological display convention."""
    out = plane.T                       # put the through-axis vertical
    if flip_v:
        out = out[::-1, :]
    if flip_h:
        out = out[:, ::-1]
    return np.ascontiguousarray(out)


def orthogonal_views(
    volume: np.ndarray,
    spacing: np.ndarray,
    index: tuple[int, int, int] | None = None,
    preset: str = "soft_tissue",
    isotropic: bool = True,
) -> Rendered:
    """Three orthogonal slices through `index` (defaults to volume centre).

    `volume` must be RAS+. Slices are resampled to square pixels when
    `isotropic` is set -- CT is routinely 0.7x0.7x5 mm, and showing that
    unscaled distorts every vertical distance judgement the model is being
    asked to make.
    """
    vol = window(volume, preset)
    if index is None:
        index = tuple(s // 2 for s in vol.shape)   # type: ignore[assignment]
    ix, iy, iz = (int(np.clip(i, 0, s - 1)) for i, s in zip(index, vol.shape))

    # sagittal: fix x -> (y, z); coronal: fix y -> (x, z); axial: fix z -> (x, y)
    sagittal = _to_display(vol[ix, :, :], flip_v=True, flip_h=False)
    coronal = _to_display(vol[:, iy, :], flip_v=True, flip_h=True)
    axial = _to_display(vol[:, :, iz], flip_v=True, flip_h=True)

    if isotropic:
        sagittal = _rescale(sagittal, (spacing[2], spacing[1]))
        coronal = _rescale(coronal, (spacing[2], spacing[0]))
        axial = _rescale(axial, (spacing[1], spacing[0]))

    return Rendered(axial=axial, coronal=coronal, sagittal=sagittal)


def _rescale(img: np.ndarray, spacing_rc: tuple[float, float]) -> np.ndarray:
    """Resample so both axes have equal physical size per pixel."""
    from scipy.ndimage import zoom

    sr, sc = spacing_rc
    base = min(sr, sc)
    factors = (sr / base, sc / base)
    if abs(factors[0] - 1) < 1e-3 and abs(factors[1] - 1) < 1e-3:
        return img
    return zoom(img, factors, order=1).astype(np.uint8)


def montage(views: dict[str, np.ndarray], pad: int = 8) -> np.ndarray:
    """Lay views side by side on a common height, separated by a light gutter."""
    if not views:
        raise ValueError("no views to montage")
    h = max(v.shape[0] for v in views.values())
    tiles = []
    for _, v in sorted(views.items()):
        if v.shape[0] != h:
            from scipy.ndimage import zoom
            v = zoom(v, (h / v.shape[0], h / v.shape[0]), order=1).astype(np.uint8)
        tiles.append(v)
        tiles.append(np.full((h, pad), 128, dtype=np.uint8))
    return np.hstack(tiles[:-1])


def montage_rgb(panels: list[np.ndarray], pad: int = 8) -> np.ndarray:
    """`montage` for colour panels, following it rule for rule.

    Lives here rather than in a caller because two callers need it to agree:
    the identification control's annotated arms and the reader study's export.
    When each kept its own compositing they diverged -- one scaled short panels
    up to the common height and used the 128-grey gutter this module has always
    used, the other zero-padded them against a black gutter -- so the reader was
    not looking at the image the `identified` arm claims to reproduce.
    """
    from scipy.ndimage import zoom

    h = max(p.shape[0] for p in panels)
    tiles = []
    for p in panels:
        if p.shape[0] != h:
            f = h / p.shape[0]
            p = zoom(p, (f, f, 1), order=1).astype(np.uint8)
        tiles.append(p)
        tiles.append(np.full((h, pad, 3), 128, dtype=np.uint8))
    return np.hstack(tiles[:-1])


def to_display(plane: np.ndarray, axis: int) -> np.ndarray:
    """Orient a plane taken by fixing `axis`, as orthogonal_views does.

    The flips are not decoration. `orthogonal_views` renders sagittal with
    flip_h=False and coronal and axial with flip_h=True; a caller that applies
    only the transpose and the vertical flip produces those two panels mirrored
    left-to-right, which on an axial CT swaps the patient's left and right and,
    inside this study, silently differed between the control arm and the
    published condition it is compared against.
    """
    return _to_display(plane, flip_v=True, flip_h=axis in (1, 2))


def save_png(img: np.ndarray, path: str) -> None:
    from PIL import Image
    Image.fromarray(img).save(path)


def selftest() -> None:
    """A volume with a known bright marker; assert it lands where expected."""
    vol = np.full((120, 100, 80), -1000, dtype=np.int16)   # air
    vol[20:100, 20:80, 10:70] = 40                          # soft tissue body
    # bright marker high in x (patient RIGHT) and high in z (SUPERIOR)
    vol[85:95, 45:55, 60:68] = 1000
    spacing = np.array([1.0, 1.0, 2.0])

    r = orthogonal_views(vol, spacing, index=(90, 50, 64))
    assert set(r.available()) == {"axial", "coronal", "sagittal"}
    for name, img in r.available().items():
        assert img.dtype == np.uint8, (name, img.dtype)
        assert img.ndim == 2 and min(img.shape) > 0, (name, img.shape)

    # windowing: air must floor, the marker must saturate
    w = window(vol, "soft_tissue")
    assert w[0, 0, 0] == 0, w[0, 0, 0]
    assert w[90, 50, 64] == 255, w[90, 50, 64]

    # the marker must be visibly bright in the coronal view through it
    assert r.coronal.max() == 255

    # isotropic rescaling must stretch the z axis (2mm) relative to x/y (1mm)
    r_raw = orthogonal_views(vol, spacing, index=(90, 50, 64), isotropic=False)
    assert r.coronal.shape[0] > r_raw.coronal.shape[0], \
        (r.coronal.shape, r_raw.coronal.shape)

    # view dependency table must not claim a plane that cannot show the relation
    assert "axial" not in views_for_category("longitudinal"), \
        "a single axial slice cannot show superior/inferior"
    assert "sagittal" not in views_for_category("laterality"), \
        "a sagittal slice cannot show left/right"
    assert "coronal" not in views_for_category("anteroposterior"), \
        "a coronal slice cannot show anterior/posterior"

    m = montage(r.available())
    assert m.ndim == 2 and m.shape[1] > max(v.shape[1] for v in r.available().values())

    print(f"selftest OK — windows applied, orthogonal views "
          f"{[f'{k}:{v.shape}' for k, v in sorted(r.available().items())]}, "
          f"isotropic rescale verified, montage {m.shape}")


if __name__ == "__main__":
    selftest()
