"""Fig.: what the volumetric models are shown.  One growth-contact probe under the four identification
conditions, rendered with the repository's own slice selection, windowing, orientation and resampling.
Outlines are drawn as anti-aliased contours of the exact resampled masks (display only; the model
receives 1-px pixel outlines).  Run from the repository root."""
import sys, os, numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D3 = os.path.join(REPO, "suite3d"); MSD = os.environ.get("MSD_ROOT", os.path.join(REPO, "data", "msd"))
OUT = os.path.join(REPO, "figures", "out"); os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, f"{D3}/spatialgen"); sys.path.insert(0, D3)
import run_identification_control as ric
from run_pipeline import label_map
from lesion_binding import LESION_LABEL, find_lesions
from scene_graph import load_ras
from render import window, to_display
sys.path.insert(0, os.path.dirname(__file__)); import house as H
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from scipy.ndimage import label as cc_label
TASK, VID, LK, TARGET = "Task10_Colon", "colon_111", "lesion1", "liver"
LES_C, TAR_C = "#ff3c3c", "#3cc8ff"
vol, affine = load_ras(f"{MSD}/{TASK}/imagesTr/{VID}.nii.gz")
gt, _ = load_ras(f"{MSD}/{TASK}/labelsTr/{VID}.nii.gz")
seg, _ = load_ras(f"{D3}/cfqa_{TASK}/seg_cache/{VID}_seg.nii.gz")
spacing = np.abs(np.diag(affine)[:3]); vol = vol.astype(np.int16)
lesion = dict(find_lesions(gt == LESION_LABEL[TASK], affine))[LK]
name2lab = {v: k for k, v in label_map().items()}; tmask = seg == name2lab[TARGET]
grey3d = window(vol, "soft_tissue")

def exact_panels(cond):
    """(grey rgb, filled lesion mask, filled target mask, iso mm/px) per panel, same slices as the model's montage."""
    _, geom = ric.render(vol, lesion, tmask, spacing, cond)
    out = []
    for a, k in zip((2, 1, 0), geom["slices"]):
        sl = [slice(None)] * 3; sl[a] = int(np.clip(k, 0, vol.shape[a] - 1))
        g, le, tg = (to_display(x[tuple(sl)], a) for x in (grey3d, lesion, tmask))
        rem = [i for i in (0, 1, 2) if i != a]
        g, le, tg, iso = ric._isotropic(g, le, tg, float(spacing[rem[1]]), float(spacing[rem[0]]))
        out.append((np.dstack([g] * 3), le, tg, iso))
    print(cond, "slices", geom["slices"], "lesion px", [int(x[1].sum()) for x in out], "target px", [int(x[2].sum()) for x in out])
    return out
def body_bbox(g, margin):
    body = g[..., 1] > 30; lab, n = cc_label(body)
    if n > 1:
        sizes = np.bincount(lab.ravel()); sizes[0] = 0; body = lab == sizes.argmax()
    ys, xs = np.nonzero(body)
    return (max(0, ys.min() - margin), min(g.shape[0], ys.max() + margin), max(0, xs.min() - margin), min(g.shape[1], xs.max() + margin))
def draw(ax, g, le, tg, iso, lw, annotate, bar, crop):
    y0, y1, x0, x1 = crop; g, le, tg = g[y0:y1, x0:x1], le[y0:y1, x0:x1], tg[y0:y1, x0:x1]
    ax.imshow(g, interpolation="none")
    if annotate:
        for m, col in ((tg, TAR_C), (le, LES_C)):
            if m.any(): ax.contour(m.astype(float), levels=[0.5], colors=[col], linewidths=lw, antialiased=True)
    if bar:
        hh = g.shape[0]; ax.add_patch(Rectangle((6, hh - 10), 10.0 / iso, 3, fc="white", ec="none"))
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values(): sp.set_edgecolor("#c9c6c1"); sp.set_linewidth(0.6)
    return g.shape

P = {cond: exact_panels(cond) for cond in ric.IMAGE_CONDITIONS}
fig = plt.figure(figsize=(7.2, 4.4)); FW, FH = fig.get_size_inches()
gs = GridSpec(1, 4, figure=fig, top=0.96, bottom=0.60, left=0.05, right=0.985, wspace=0.06)
TITLES = [("plain", "centre slices\nno annotation"), ("bestslice", "slices on which both\nstructures are visible"),
          ("overlay", "centre slices\nlegend names a red outline"), ("identified", "joint-visibility slices\noutlines and a 10 mm bar")]
for k, (cond, desc) in enumerate(TITLES):
    g, le, tg, iso = P[cond][0]                                   # the axial panel
    ax = fig.add_subplot(gs[0, k])
    draw(ax, g, le, tg, iso, 0.75, annotate=cond in ("overlay", "identified"), bar=cond == "identified", crop=body_bbox(g, 10))
    ax.set_title(f"({'abcd'[k]})  {cond}", fontsize=7.2, loc="left", pad=3)
    ax.text(0.5, -0.05, desc, transform=ax.transAxes, ha="center", va="top", fontsize=6.0, color=H.CHARCOAL, linespacing=1.15)
# row (e): the three identified panels at ONE physical scale, top-aligned; the question under the shorter axial panel
views = [(g, le, tg, iso, body_bbox(g, 6)) for g, le, tg, iso in P["identified"]]
w_mm = [(c[3] - c[2]) * iso for g, le, tg, iso, c in views]; h_mm = [(c[1] - c[0]) * iso for g, le, tg, iso, c in views]
GAP = 0.08; LEFT, RIGHT = 0.05 * FW, 0.985 * FW; ROW_TOP, ROW_BOT = 0.475 * FH, 0.035 * FH
scale = min((RIGHT - LEFT - 2 * GAP) / sum(w_mm), (ROW_TOP - ROW_BOT) / max(h_mm))
x = LEFT + ((RIGHT - LEFT) - (sum(w_mm) * scale + 2 * GAP)) / 2
fig.text(x / FW, (ROW_TOP + 0.06) / FH, "(e)  identified, as the model receives it: the three panels at one physical scale, cropped to the body", fontsize=7.2, ha="left", va="bottom")
for k, ((g, le, tg, iso, crop), name) in enumerate(zip(views, ["axial", "coronal", "sagittal"])):
    w_in, h_in = w_mm[k] * scale, h_mm[k] * scale
    ax = fig.add_axes([x / FW, (ROW_TOP - h_in) / FH, w_in / FW, h_in / FH])
    draw(ax, g, le, tg, iso, 0.9, annotate=True, bar=True, crop=crop)
    ax.text(0.03, 0.96, name, transform=ax.transAxes, ha="left", va="top", fontsize=6.4, color="white", bbox=dict(boxstyle="round,pad=0.25", fc="black", ec="none", alpha=0.55))
    x += w_in + GAP
fig.savefig(os.path.join(OUT, "fig_render.pdf"), dpi=450); plt.close(fig); print("wrote figures/out/fig_render.pdf")
