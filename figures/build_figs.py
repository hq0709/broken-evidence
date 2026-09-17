"""Build the decision-variable, four-arm, sub-task and 2D decay figures from the result files.
Run from the repository root:  python figures/build_figs.py
"""
import csv, json, sys, os, numpy as np, collections
sys.path.insert(0, os.path.dirname(__file__)); import house as H
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D3 = os.path.join(REPO, "suite3d")
OUT = os.path.join(REPO, "figures", "out"); os.makedirs(OUT, exist_ok=True)
R2D = os.path.join(REPO, "results", "2d", "blur_aggregate.csv")
LEG = dict(fontsize=6.3, handlelength=1.4, borderpad=0.35, labelspacing=0.3, handletextpad=0.4, columnspacing=1.0)
INK = "#3a3a3a"; MUTED = "#6a6a6a"

# ================= Fig. decision: (a) grouped bars, three AUROCs per model  (b) scale ladder =================
CH = "#9a9a9a"
DV = list(csv.DictReader(open(os.path.join(REPO, "results", "3d", "decision_variable.csv"))))
for r in DV:
    for k in list(r):
        if k.startswith("auc") or k in ("acc", "params_b"): r[k] = float(r[k])
main = sorted([r for r in DV if r["arm"] == "sighted"], key=lambda r: -r["auc_pair"])
fig, (a, c) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.9, 1.0]})
H.panel_title(a, "a", "AUROC of the decision variable, thirteen models"); a.grid(False, axis="x")
X = np.arange(len(main)); w = 0.26; gap = 0.02
SER = [("auc_pair", "growth number (same volume, number differs)", H.LILAC), ("auc_gold", "computed label (all probes)", H.CHARCOAL), ("auc_text", "volume (same sentence, volume differs)", H.SAGE)]
hs = []
for k, (key, lab, col) in enumerate(SER):
    xs = X + (k - 1) * (w + gap); ys = [r[key] for r in main]
    lo = [r[key] - r[key + "_lo"] for r in main]; hi = [r[key + "_hi"] - r[key] for r in main]
    hs.append(a.bar(xs, ys, width=w, color=col, edgecolor="none", label=lab, zorder=3))
    a.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor="#555", elinewidth=0.6, capsize=1.3, zorder=4)
a.axhline(0.5, color=CH, lw=0.8, zorder=2); a.annotate("chance", (len(main) - 0.55, 0.5), xytext=(0, 2), textcoords="offset points", ha="right", va="bottom", fontsize=6.2, color=MUTED)
a.set_xticks(X); a.set_xticklabels([r["model"].replace("-OneVision", "-OV").replace("M3D-LaMed-", "M3D-") + ("$^{\\dagger}$" if r["input"] == "native" else "") for r in main], fontsize=6.2, rotation=35, ha="right", rotation_mode="anchor")
a.set_xlim(-0.6, len(main) - 0.4); a.set_ylim(0.3, 1.0); a.set_yticks([0.3, 0.5, 0.7, 0.9]); a.set_ylabel("AUROC of $\\log p_{\\rm yes}-\\log p_{\\rm no}$", fontsize=7.8)
H.boxed_legend(a, hs, loc="upper right", fontsize=6.0, handlelength=1.2, borderpad=0.4, labelspacing=0.3, handletextpad=0.5)
# (b) scale ladder, Qwen2.5-VL family on the plain rendering
lad = {r["tag"]: r for r in DV if r["arm"] in ("sighted", "plain") and r["tag"] in ("qwen3b", "qwen7b", "qwen32b", "qwen72b")}
xs = [lad[t]["params_b"] for t in ("qwen3b", "qwen7b", "qwen32b", "qwen72b")]
H.panel_title(c, "b", "Scale reads the number, not the label")
for k, lab_, col, mk in [("auc_pair", "growth number", H.LILAC, "D"), ("auc_gold", "computed label", H.CHARCOAL, "o")]:
    ys = [lad[t][k] for t in ("qwen3b", "qwen7b", "qwen32b", "qwen72b")]; lo = [lad[t][k + "_lo"] for t in ("qwen3b", "qwen7b", "qwen32b", "qwen72b")]; hi = [lad[t][k + "_hi"] for t in ("qwen3b", "qwen7b", "qwen32b", "qwen72b")]
    c.fill_between(xs, lo, hi, color=col, alpha=0.14, lw=0, zorder=1); c.plot(xs, ys, ls="--", lw=1.0, color=col, marker=mk, ms=4.5, mec="white", mew=0.6, zorder=3, label=lab_)
    for x, y in zip(xs, ys): H.value_label(c, x, y, f"{y:.2f}", col, dx=0, dy=6 if k == "auc_pair" else -9, ha="center", va="bottom" if k == "auc_pair" else "top", size=6.0)
c.axhline(0.5, color=CH, lw=0.8, zorder=1); c.annotate("chance", (15, 0.5), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=6.2, color=MUTED)
c.set_xscale("log"); c.set_xticks(xs); c.set_xticklabels(["3B", "7B", "32B", "72B"]); c.minorticks_off(); c.set_xlim(2.2, 100)
c.set_ylim(0.35, 1.04); c.set_yticks([0.4, 0.6, 0.8, 1.0]); c.set_xlabel("Qwen2.5-VL parameters", fontsize=8); c.set_ylabel("AUROC", fontsize=8)
H.boxed_legend(c, c.get_legend_handles_labels()[0], loc="center right", fontsize=6.2, handlelength=1.4, borderpad=0.4, labelspacing=0.35)
fig.tight_layout(w_pad=1.6); fig.savefig(os.path.join(OUT, "fig_decision.pdf"), bbox_inches="tight"); plt.close(fig)

# ================= Fig. 5: row 1 accuracy, row 2 modal share =================
rows = list(csv.DictReader(open(f"{D3}/figdata/fig9_roi_four_arm.csv"))); organs = ["Lung", "Colon", "Pancreas", "Liver"]
arms = [("full", "full volume", H.HAZE, "o"), ("roi_masked", "evidence region removed", H.ROSE, "s"), ("roi_only", "only evidence region kept", H.OAT, "^")]
def get(model, o, arm, k): return float(next(r[k] for r in rows if r["model"] == model and r["organ"] == o and r["arm"] == arm))
fig, axes = plt.subplots(2, 2, figsize=(7.0, 3.9), sharex=True, gridspec_kw={"height_ratios": [1.15, 1]})
for j, (letter, model) in enumerate([("a", "Qwen2.5-VL-32B"), ("b", "InternVL3-8B")]):
    ax = axes[0, j]; H.panel_title(ax, letter, f"{model}: accuracy")
    for k, (arm, label, col, mk) in enumerate(arms):
        xs = [i + (k - 1) * 0.24 for i in range(len(organs))]; ys = [get(model, o, arm, "acc") for o in organs]
        lo = [ys[i] - get(model, o, arm, "ci_lo") for i, o in enumerate(organs)]; hi = [get(model, o, arm, "ci_hi") - ys[i] for i, o in enumerate(organs)]
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt=mk, ms=5.5, mfc=col, mec="white", mew=0.9, ecolor=col, elinewidth=1.2, capsize=2, capthick=0.9, zorder=3)
    ax.axhline(50, color=H.GREY, ls="--", lw=1.0, zorder=2); ax.set_ylim(46, 66); ax.set_yticks([50, 55, 60, 65]); ax.set_xlim(-0.6, 3.6)
    ax2 = axes[1, j]; H.panel_title(ax2, chr(ord(letter) + 2), f"{model}: modal answer share")
    for k, (arm, label, col, mk) in enumerate(arms):
        xs = [i + (k - 1) * 0.24 for i in range(len(organs))]; ys = [get(model, o, arm, "modal_share") for o in organs]
        ax2.bar(xs, ys, width=0.22, color=col, edgecolor="none", zorder=3)
        for x, y in zip(xs, ys):
            if arm == "roi_only": H.value_label(ax2, x, y, f"{y:.0f}", col, dy=1.5, size=6.2)
    ax2.axhline(100, color=H.GREY, ls=":", lw=0.8); ax2.set_ylim(40, 104); ax2.set_yticks([50, 75, 100]); ax2.set_xticks(range(len(organs))); ax2.set_xticklabels(organs); ax2.grid(False, axis="x")
axes[0, 0].set_ylabel("Accuracy (%)"); axes[1, 0].set_ylabel("Modal share (%)")
axes[0, 0].annotate("blank volume = 50.0", (1.5, 49.5), ha="center", va="top", fontsize=6.6, color=MUTED)
H.boxed_legend(axes[0, 0], [H.series_handle(c, m, l, ls="none") for _, l, c, m in arms], loc="upper right", ncol=1, **LEG)
fig.tight_layout(h_pad=0.8, w_pad=1.0); fig.savefig(os.path.join(OUT, "fig_four_arm.pdf"), bbox_inches="tight"); plt.close(fig)

# ================= Fig. 6: (a) sub-task bars  (b) distance-answer distribution =================
models = ["Qwen2.5-VL-7B", "InternVL3-8B", "Qwen3-VL-8B", "Qwen2.5-VL-32B"]; tags = ["qwen7b", "internvl", "qwen3vl", "qwen32b"]
xlab = ["Qwen2.5-VL\n7B", "InternVL3\n8B", "Qwen3-VL\n8B", "Qwen2.5-VL\n32B"]
tasks = [("localise, annotated", H.HAZE), ("localise, unannotated", H.LILAC), ("name target", H.SAGE), ("distance (mm)", H.ROSE)]
acc = {"Qwen2.5-VL-7B": [67.7, 57.3, 34.3, 25.7], "InternVL3-8B": [41.0, 15.0, 31.3, 26.3], "Qwen3-VL-8B": [81.7, 49.7, 60.0, 23.0], "Qwen2.5-VL-32B": [76.0, 52.0, 45.0, 28.3]}
ci = {"Qwen2.5-VL-7B": [(61.1, 74.1), (50.0, 64.3), (29.4, 39.5), (20.7, 31.0)], "InternVL3-8B": [(34.1, 48.2), (10.8, 19.6), (26.3, 36.5), (21.6, 31.3)],
      "Qwen3-VL-8B": [(76.4, 86.6), (42.5, 56.8), (54.3, 65.6), (18.3, 27.9)], "Qwen2.5-VL-32B": [(70.4, 81.4), (44.6, 59.2), (39.3, 51.1), (23.1, 33.8)]}
BUCK = ["5", "15", "25", "35"]; dist = {}
for m, t in zip(models, tags):
    R = [json.loads(l) for l in open(f"{D3}/results_new/id_common_{t}_subtask-distance.jsonl")]
    c = collections.Counter(r["prediction"] for r in R); dist[m] = [c.get(k, 0) for k in BUCK]
    if m == models[0]:
        g = collections.Counter(r["gold"] for r in R); dist["gold"] = [g.get(k, 0) for k in BUCK]
fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.25, 1]})
H.panel_title(a, "a", "Four sub-tasks on the identified rendering  (chance 25%)"); a.grid(False, axis="x")
X = np.arange(len(models)); w = 0.19; gap = 0.015; hs = []
for j, (name, col) in enumerate(tasks):
    xs = X + (j - 1.5) * (w + gap); ys = [acc[m][j] for m in models]
    lo = [acc[m][j] - ci[m][j][0] for m in models]; hi = [ci[m][j][1] - acc[m][j] for m in models]
    hs.append(a.bar(xs, ys, width=w, color=col, edgecolor="none", label=name, zorder=3))
    a.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor="#555", elinewidth=0.7, capsize=1.5, zorder=4)
hs.append(a.axhline(25, color="#9a9a9a", lw=0.8, ls="-", zorder=2, label="chance 25%"))
for i, m in enumerate(models): H.value_label(a, X[i] + (3 - 1.5) * (w + gap), ci[m][3][1], f"{acc[m][3]:.1f}", H.ROSE, dy=2, size=7)
a.set_xticks(X); a.set_xticklabels(xlab, fontsize=7); a.set_xlim(-0.6, 3.6); a.set_ylim(0, 100); a.set_yticks([0, 25, 50, 75, 100]); a.set_ylabel("Accuracy (%)")
H.boxed_legend(a, hs, loc="upper center", ncol=1, bbox_to_anchor=(0.36, 0.99), fontsize=6.0, handlelength=1.2, borderpad=0.35, labelspacing=0.3, handletextpad=0.4)
H.panel_title(b, "b", "Distance answers: where the 300 items land"); b.grid(False, axis="y")
rowsb = [("gold", dist["gold"])] + [(m, dist[m]) for m in models]
shades = [H.HAZE, H.SAGE, H.OAT, H.ROSE]
for i, (name, d) in enumerate(rowsb):
    left = 0; tot = sum(d)
    for k, (v, colk) in enumerate(zip(d, shades)):
        b.barh(i, v, left=left, color=colk, edgecolor="white", linewidth=0.6, height=0.62, alpha=1.0 if name != "gold" else 0.55, zorder=3)
        if v / tot >= 0.12: b.annotate(f"{v}", (left + v / 2, i), ha="center", va="center", fontsize=6.4, color="white" if name != "gold" else "#333", fontweight="bold")
        left += v
b.set_yticks(range(len(rowsb))); b.set_yticklabels(["gold (n=300)"] + models, fontsize=6.6); b.invert_yaxis()
b.set_xlim(0, 300); b.set_xticks([0, 100, 200, 300]); b.set_xlabel("items", fontsize=8)
hb = [Line2D([], [], marker="s", ms=7, color=c, ls="none", label=f"{k} mm") for k, c in zip(BUCK, shades)]
H.boxed_legend(b, hb, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.42), **LEG)
fig.tight_layout(w_pad=1.4); fig.savefig(os.path.join(OUT, "fig_subtask.pdf"), bbox_inches="tight"); plt.close(fig)

# ================= Fig. decay (2D only, single column) =================
SIG = [0, 2, 4, 8, 16, 32, 64]; XP = list(range(len(SIG))) + [len(SIG) + 0.8]; XINF = XP[-1]; CH = "#9a9a9a"
R2 = list(csv.DictReader(open(R2D)))
def acc2d(model, group, s):
    key = "inf" if s == "inf" else str(s)
    return 100 * float(next(r["acc"] for r in R2 if r["model_id"] == model and r["group"] == group and r["tier"] == "all" and r["sigma"] == key))
m2d = [("gpt-4o", "GPT-4o", 64, H.HAZE, "o"), ("claude-sonnet-4-6", "Claude Sonnet 4.6", 32, H.ROSE, "s"),
       ("Qwen--Qwen3.5-397B-A17B", "Qwen3.5-397B", 16, H.LILAC, "^"), ("huatuogpt-vision-7b", "HuatuoGPT-V 7B", 32, H.OAT, "D")]
fig, ax = plt.subplots(figsize=(3.5, 2.7)); hs = []
for mid, name, L, col, mk in m2d:
    y = [acc2d(mid, "image_required", s) for s in SIG] + [acc2d(mid, "image_required", "inf")]
    ax.plot(XP[:-1], y[:-1], ls="--", lw=1.0, color=col, marker=mk, ms=4.2, mec="white", mew=0.6, zorder=3)
    ax.plot(XINF, y[-1], marker="x", ms=6, mew=1.5, color=col, mec=col, ls="none", zorder=3)
    hs.append(H.series_handle(col, mk, f"{name} ($L^\\star$={L})"))
hs.append(Line2D([], [], marker="x", ms=6, mew=1.5, color="#666", mec="#666", ls="none", label="no image"))
ax.axhline(20, color=CH, lw=0.8, zorder=2); ax.annotate("chance", (-0.45, 20), xytext=(1, 2), textcoords="offset points", ha="left", va="bottom", fontsize=6.3, color=MUTED)
ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 100); ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.set_xticks(XP); ax.set_xticklabels([str(s) for s in SIG] + ["$\\infty$"], fontsize=7); ax.set_xlim(-0.5, XINF + 0.6); ax.set_xlabel("Blur $\\sigma$ (px)")
H.boxed_legend(ax, hs, loc="upper right", fontsize=6.0, handlelength=1.6, borderpad=0.35, labelspacing=0.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_decay.pdf"), bbox_inches="tight"); plt.close(fig)
print("built figures")
