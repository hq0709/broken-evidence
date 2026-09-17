"""Fig.: two ways the evidence contract breaks in the volume.
(a) one item on three backgrounds  (b) distance accuracy per model x background
(c) modal answer and its share per model x background (answers are habits, not readings)
(d) the same comparison in three forms: numbers only, numbers inside the clinical sentence, the image.
Run after suite3d/analysis/metric_control.py."""
import csv, os, sys, json, glob, collections, numpy as np
sys.path.insert(0, os.path.dirname(__file__)); import house as H
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from scipy.ndimage import binary_dilation
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D3 = os.path.join(REPO, "suite3d")
OUT = os.path.join(REPO, "figures", "out"); os.makedirs(OUT, exist_ok=True)
ITEM = ("Task10_Colon", "colon_022_lesion1_duodenum")
MODELS = [("qwen7b", "Qwen2.5-VL-7B"), ("internvl", "InternVL3-8B"), ("qwen3vl", "Qwen3-VL-8B"), ("qwen32b", "Qwen2.5-VL-32B")]
SHORT = ["Qwen2.5-VL\n7B", "InternVL3\n8B", "Qwen3-VL\n8B", "Qwen2.5-VL\n32B"]
BG = [("grey", "synthetic outlines, grey", H.SAGE), ("ct", "synthetic outlines on CT", H.OAT), ("real", "real lesion and target", H.ROSE)]
rows = list(csv.DictReader(open(f"{D3}/figdata/metric_control.csv")))
for r in rows:
    for k in ("acc", "lo", "hi", "modal_share"): r[k] = float(r[k])
def cell(m, bg): return next(r for r in rows if r["model"] == m and r["background"] == bg)
# (d) data: numeric oracle, text oracle, image (published plain rendering), growth-matched subset
def acc(files):
    R = [json.loads(l) for f in files for l in open(f)]
    return 100 * np.mean([r["prediction"] == r["gold"] for r in R]), collections.Counter(r["prediction"] for r in R).most_common(1)[0]
FORMS = [("numeric", "numbers only", H.CHARCOAL), ("text", "numbers inside the clinical sentence", H.LILAC), ("image", "the image (published rendering)", H.HAZE)]
D = {}
for tag, name in MODELS:
    D[(tag, "numeric")] = acc(glob.glob(f"{D3}/results_new/id_common_{tag}_numeric-oracle.jsonl"))
    D[(tag, "text")] = acc(glob.glob(f"{D3}/results_new/id_common_{tag}_text-oracle.jsonl"))
    D[(tag, "image")] = acc(glob.glob(f"{D3}/results_new/id_Task*_{tag}_plain.jsonl"))

fig = plt.figure(figsize=(7.2, 5.3))
outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.95], hspace=0.42)
top = outer[0].subgridspec(3, 2, width_ratios=[0.5, 1.45], wspace=0.04, hspace=0.25)
bot = outer[1].subgridspec(1, 2, width_ratios=[0.95, 1.2], wspace=0.42)

# ---- (a) thumbnails
def axial(path):
    im = np.array(Image.open(path).convert("RGB")); gut = np.all(im == 128, axis=(0, 2)); x = int(np.argmax(gut)) if gut.any() else im.shape[1]
    return im[:, :x]
for i, (bg, lab, col) in enumerate(BG):
    ax = fig.add_subplot(top[i, 0])
    p = f"{D3}/render_cache/synthetic/{bg}/{ITEM[1]}.png" if bg != "real" else f"{D3}/render_cache/{ITEM[0]}/{ITEM[1]}_slices3.png"
    im = axial(p); h, w = im.shape[:2]
    ann = (np.abs(im[..., 0].astype(int) - im[..., 1].astype(int)) > 40) | (np.abs(im[..., 1].astype(int) - im[..., 2].astype(int)) > 40); ann[h - 12:, :] = False
    ys, xs = np.nonzero(ann); m = 45
    y0, y1, x0, x1 = max(0, ys.min() - m), min(h, ys.max() + m), max(0, xs.min() - m), min(w, xs.max() + m)
    side = max(y1 - y0, x1 - x0); cy, cx = (y0 + y1) // 2, (x0 + x1) // 2; y0, x0 = max(0, cy - side // 2), max(0, cx - side // 2)
    crop = im[y0:y0 + side, x0:x0 + side].copy(); ca = ann[y0:y0 + side, x0:x0 + side]
    for col_, mask in ((np.array([255, 60, 60]), ca & (crop[..., 0] > crop[..., 2])), (np.array([60, 200, 255]), ca & (crop[..., 2] > crop[..., 0]))):
        crop[binary_dilation(mask, iterations=2)] = col_
    ax.imshow(crop, interpolation="bilinear"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values(): sp.set_edgecolor(col); sp.set_linewidth(1.2)
    ax.set_ylabel(lab, fontsize=6.4, color=H.CHARCOAL, rotation=0, ha="right", va="center", labelpad=6)
fig.text(0.075, 0.905, "(a)  one item, three backgrounds", fontsize=8, ha="left", va="bottom")

# ---- (b) accuracy bars
ax = fig.add_subplot(top[:, 1]); H.panel_title(ax, "b", "Distance sub-task accuracy  (chance 25%)"); ax.grid(False, axis="x")
X = np.arange(len(MODELS)); w = 0.26; gap = 0.02; hs = []
for k, (bg, lab, col) in enumerate(BG):
    ys = [cell(n, bg)["acc"] for _, n in MODELS]; lo = [cell(n, bg)["acc"] - cell(n, bg)["lo"] for _, n in MODELS]; hi = [cell(n, bg)["hi"] - cell(n, bg)["acc"] for _, n in MODELS]
    xs = X + (k - 1) * (w + gap); hs.append(ax.bar(xs, ys, width=w, color=col, edgecolor="none", label=lab, zorder=3))
    ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor="#555", elinewidth=0.7, capsize=1.5, zorder=4)
ax.axhline(25, color="#9a9a9a", lw=0.8, zorder=2)
ax.set_xticks(X); ax.set_xticklabels(SHORT, fontsize=6.8); ax.set_xlim(-0.6, len(MODELS) - 0.4); ax.set_ylim(0, 100); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_ylabel("Accuracy (%)", fontsize=8)
H.boxed_legend(ax, hs, loc="upper right", fontsize=6.2, handlelength=1.2, borderpad=0.4, labelspacing=0.3)

# ---- (c) modal answer heatmap
ax = fig.add_subplot(bot[0, 0]); H.panel_title(ax, "c", "The answer is a habit: modal answer and its share"); ax.grid(False)
M = np.array([[cell(n, bg)["modal_share"] for bg, *_ in BG] for _, n in MODELS])
cmap = LinearSegmentedColormap.from_list("share", ["#F5F1EC", "#E3C9C5", H.ROSE], N=256)
ax.imshow(M, cmap=cmap, vmin=40, vmax=100, aspect="auto")
for i in range(len(MODELS)):
    for j, (bg, *_) in enumerate(BG):
        c = cell(MODELS[i][1], bg); ax.text(j, i, f"\u201c{c['modal']}\u201d\n{c['modal_share']:.0f}%", ha="center", va="center", fontsize=6.6, color="#2f2f2f" if c["modal_share"] < 85 else "white")
ax.set_xticks(range(3)); ax.set_xticklabels(["grey", "CT", "real"], fontsize=7); ax.set_yticks(range(len(MODELS))); ax.set_yticklabels([n for _, n in MODELS], fontsize=6.8)
ax.set_xlabel("background", fontsize=7.5); ax.tick_params(length=0)
for sp in ax.spines.values(): sp.set_visible(False)

# ---- (d) the same comparison in three forms
ax = fig.add_subplot(bot[0, 1]); H.panel_title(ax, "d", "The same comparison, three forms  (chance 50%)"); ax.grid(False, axis="x")
hs = []
for k, (form, lab, col) in enumerate(FORMS):
    ys = [D[(t, form)][0] for t, _ in MODELS]; xs = X + (k - 1) * (w + gap)
    hs.append(ax.bar(xs, ys, width=w, color=col, edgecolor="none", label=lab, zorder=3))
    for x, y, (t, _) in zip(xs, ys, MODELS):
        H.value_label(ax, x, y, f"{y:.0f}", col, dy=2, size=6.0)
ax.axhline(50, color="#9a9a9a", lw=0.8, zorder=2)
ax.set_xticks(X); ax.set_xticklabels(SHORT, fontsize=6.8); ax.set_xlim(-0.6, len(MODELS) - 0.4); ax.set_ylim(0, 150); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_ylabel("Accuracy (%)", fontsize=8)
H.boxed_legend(ax, hs, loc="upper right", fontsize=6.0, handlelength=1.2, borderpad=0.4, labelspacing=0.3)
fig.savefig(os.path.join(OUT, "fig_metric.pdf"), bbox_inches="tight"); plt.close(fig); print("wrote figures/out/fig_metric.pdf")
for t, n in MODELS: print(n, {f: (round(D[(t, f)][0], 1), D[(t, f)][1]) for f, *_ in FORMS})
