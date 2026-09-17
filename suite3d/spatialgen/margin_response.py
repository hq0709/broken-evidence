"""
Margin-response analysis: does a model's accuracy track the geometric difficulty
of each item?

The instrument
--------------
Every directional QA this pipeline generates carries `margin_mm` -- the exact
separation, in millimetres, between the two structures along the queried axis.
That is a continuous, physically computed difficulty parameter, known without
error, and not a proxy: a 2 mm separation is genuinely harder to read off an
image than an 80 mm one.

This yields an item-level test of visual grounding that a blind baseline cannot
give. A model that actually reads the geometry MUST degrade as the margin
shrinks. A model answering from priors has no access to the margin, so its
accuracy is flat in it. Aggregate accuracy cannot distinguish those two; the
slope can.

Negative control
----------------
Run it on a text-only model with no image. The slope must be ~0. If it is not,
margin is correlated with some linguistic cue (e.g. distant organ pairs are also
the ones with famous textbook relations), and that confound has to be handled
before the instrument means anything. Running this control BEFORE claiming
anything about a VLM is the whole point.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def load_margins(qa_dir: Path) -> dict[str, dict]:
    """qid -> {margin_mm, category, gold} for items that carry a margin."""
    out = {}
    files = sorted(qa_dir.glob("*.jsonl")) if qa_dir.is_dir() else [qa_dir]
    for f in files:
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            prov = r.get("provenance") or {}
            rel = prov.get("relation") or {}
            m = rel.get("margin_mm", prov.get("margin_mm"))
            if m is None:
                continue
            out[r["qid"]] = {"margin_mm": abs(float(m)),
                             "category": r.get("category"),
                             "gold": r["answer"]}
    return out


def load_records(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def logistic_slope(xs: list[float], ys: list[int], iters: int = 400
                   ) -> tuple[float, float]:
    """Fit P(correct) = sigmoid(a + b*log10(margin)) by plain gradient ascent.

    Returns (b, mean_abs_gradient_at_end). b is the quantity of interest: the
    log-odds change in accuracy per decade of margin. b ~ 0 means the model's
    correctness is independent of geometric difficulty.
    Implemented directly to avoid a scikit-learn dependency and to keep the fit
    inspectable.
    """
    if len(xs) < 20:
        return float("nan"), float("nan")
    mx = sum(xs) / len(xs)
    xc = [x - mx for x in xs]          # centre for conditioning
    a = b = 0.0
    lr = 0.5
    for _ in range(iters):
        ga = gb = 0.0
        for x, y in zip(xc, ys):
            p = 1.0 / (1.0 + math.exp(-(a + b * x)))
            e = y - p
            ga += e
            gb += e * x
        n = len(xc)
        a += lr * ga / n
        b += lr * gb / n
    return b, abs(gb / len(xc))


def analyse(records: list[dict], margins: dict[str, dict],
            n_bins: int = 5) -> dict:
    rows = []
    for r in records:
        m = margins.get(r["qid"])
        if m is None or r.get("pred") is None:
            continue
        rows.append({"margin": m["margin_mm"], "category": m["category"],
                     "correct": int(r["pred"] == r["gold"])})
    if not rows:
        return {"n": 0}

    rows.sort(key=lambda d: d["margin"])
    per_bin = max(1, len(rows) // n_bins)
    bins = []
    for i in range(0, len(rows), per_bin):
        chunk = rows[i:i + per_bin]
        if len(chunk) < 5:
            break
        bins.append({
            "margin_lo": round(chunk[0]["margin"], 1),
            "margin_hi": round(chunk[-1]["margin"], 1),
            "n": len(chunk),
            "accuracy": sum(c["correct"] for c in chunk) / len(chunk),
        })

    xs = [math.log10(max(r["margin"], 0.1)) for r in rows]
    ys = [r["correct"] for r in rows]
    slope, grad = logistic_slope(xs, ys)

    by_cat = {}
    cats = defaultdict(list)
    for r in rows:
        cats[r["category"]].append(r)
    for c, rs in cats.items():
        if len(rs) < 20:
            continue
        s, _ = logistic_slope(
            [math.log10(max(r["margin"], 0.1)) for r in rs],
            [r["correct"] for r in rs])
        by_cat[c] = {"n": len(rs), "slope_per_decade": round(s, 3),
                     "accuracy": sum(r["correct"] for r in rs) / len(rs)}

    return {"n": len(rows), "bins": bins,
            "slope_per_decade": round(slope, 3),
            "final_gradient": round(grad, 5),
            "per_category": by_cat}


def paired(qa_dir: Path, sighted: Path, blind: Path, n_bins: int = 5) -> dict:
    """Slope excess of a sighted model over the SAME model run blind.

    Measured, not assumed: on this pipeline's QA a text-only model with no image
    still shows slope +0.441, because geometric separation is confounded with
    textbook-knowability -- widely separated organ pairs are also the pairs whose
    relation is common knowledge. So a positive sighted slope proves nothing on
    its own.

    The blind run is the null. Only the EXCESS, sighted minus blind, is
    attributable to reading the image. Reporting a raw sighted slope without this
    subtraction would credit the model for a property of the question set.
    """
    margins = load_margins(qa_dir)
    s = analyse(load_records(sighted), margins, n_bins)
    b = analyse(load_records(blind), margins, n_bins)
    if not s["n"] or not b["n"]:
        return {"error": "one of the runs has no scorable items"}

    excess = s["slope_per_decade"] - b["slope_per_decade"]
    per_cat = {}
    for c in set(s["per_category"]) & set(b["per_category"]):
        per_cat[c] = {
            "sighted": s["per_category"][c]["slope_per_decade"],
            "blind": b["per_category"][c]["slope_per_decade"],
            "excess": round(s["per_category"][c]["slope_per_decade"]
                            - b["per_category"][c]["slope_per_decade"], 3),
            "n": min(s["per_category"][c]["n"], b["per_category"][c]["n"]),
        }
    return {"sighted_slope": s["slope_per_decade"],
            "blind_slope": b["slope_per_decade"],
            "excess_slope": round(excess, 3),
            "sighted_accuracy": round(
                sum(x["accuracy"] * x["n"] for x in s["bins"])
                / max(sum(x["n"] for x in s["bins"]), 1), 4),
            "blind_accuracy": round(
                sum(x["accuracy"] * x["n"] for x in b["bins"])
                / max(sum(x["n"] for x in b["bins"]), 1), 4),
            "per_category": per_cat}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", required=True)
    ap.add_argument("--records", required=False)
    ap.add_argument("--sighted", help="records from the model WITH the image")
    ap.add_argument("--blind", help="records from the same model WITHOUT the image")
    ap.add_argument("--label", default="model")
    ap.add_argument("--bins", type=int, default=5)
    args = ap.parse_args()

    if args.sighted and args.blind:
        r = paired(Path(args.qa), Path(args.sighted), Path(args.blind), args.bins)
        if "error" in r:
            print(r["error"]); return
        print("=== paired margin-response (sighted vs blind) ===")
        print(f"  sighted slope {r['sighted_slope']:+.3f}  "
              f"(accuracy {r['sighted_accuracy']:.1%})")
        print(f"  blind   slope {r['blind_slope']:+.3f}  "
              f"(accuracy {r['blind_accuracy']:.1%})   <- confound null")
        print(f"  EXCESS  slope {r['excess_slope']:+.3f}   "
              f"<- the only part attributable to seeing the image")

        # An aggregate excess is meaningless when the per-category excesses
        # disagree in sign: averaging +0.484 against -0.436 produced a spurious
        # "+0.177 grounding" verdict on the first real run. Require the effect to
        # hold in the SAME DIRECTION across categories before claiming anything.
        cats = r["per_category"]
        pos = [c for c, v in cats.items() if v["excess"] >= 0.15]
        neg = [c for c, v in cats.items() if v["excess"] <= -0.15]
        if not cats:
            verdict = "no per-category breakdown available"
        elif r["excess_slope"] < 0.15:
            verdict = "no evidence the model reads the geometry"
        elif neg:
            verdict = (f"INCONCLUSIVE: excess is positive overall but NEGATIVE in "
                       f"{neg}. Sign disagreement across categories means the "
                       f"aggregate is an averaging artefact, not grounding.")
        elif len(pos) < max(1, len(cats) - 1):
            verdict = (f"WEAK: excess carried by {pos} only; not consistent "
                       f"across categories")
        else:
            verdict = "evidence of genuine geometric grounding"
        print(f"  -> {verdict}")

        if r["blind_accuracy"] > r["sighted_accuracy"]:
            print(f"  !! the BLIND run is more accurate than the sighted one "
                  f"({r['blind_accuracy']:.1%} vs {r['sighted_accuracy']:.1%}) -- "
                  f"the image is actively hurting this model")
        for c, v in sorted(r["per_category"].items()):
            print(f"    {c:18} sighted {v['sighted']:+.3f}  blind {v['blind']:+.3f}"
                  f"  excess {v['excess']:+.3f}  n={v['n']}")
        return

    if not args.records:
        ap.error("provide --records, or both --sighted and --blind")

    margins = load_margins(Path(args.qa))
    records = load_records(Path(args.records))
    res = analyse(records, margins, args.bins)

    print(f"=== margin-response: {args.label} ===")
    print(f"items with a computed margin: {res['n']}")
    if not res["n"]:
        return
    print(f"\n  margin range (mm)      n     accuracy")
    for b in res["bins"]:
        bar = "#" * int(round(b["accuracy"] * 30))
        print(f"  {b['margin_lo']:7.1f} - {b['margin_hi']:7.1f}  {b['n']:4d}   "
              f"{b['accuracy']:6.1%}  {bar}")

    s = res["slope_per_decade"]
    print(f"\n  logistic slope per decade of margin: {s:+.3f}")
    if abs(s) < 0.15:
        print("  -> FLAT: correctness is independent of geometric difficulty.")
        print("     For a blind model this is the expected negative control.")
        print("     For a VLM it means the answers are not read off the image.")
    else:
        direction = "easier at larger margin" if s > 0 else "harder at larger margin"
        print(f"  -> DEPENDENT ({direction}): accuracy tracks the geometry.")
        if s < 0:
            print("     NOTE: a negative slope is not a grounding signal; it "
                  "suggests margin is confounded with something else.")

    if res["per_category"]:
        print("\n  per category:")
        for c, v in sorted(res["per_category"].items()):
            print(f"    {c:18} slope {v['slope_per_decade']:+.3f}  "
                  f"acc {v['accuracy']:5.1%}  n={v['n']}")


def selftest() -> None:
    # a model that reads geometry: correctness rises with margin
    grounded = []
    margins = {}
    for i in range(400):
        m = 0.5 * (1.05 ** i)
        qid = f"q{i}"
        margins[qid] = {"margin_mm": m, "category": "longitudinal", "gold": "superior"}
        p = 1 / (1 + math.exp(-(math.log10(m) - 0.8) * 3))
        grounded.append({"qid": qid, "gold": "superior",
                         "pred": "superior" if (i * 7919) % 1000 < p * 1000 else "inferior"})
    r = analyse(grounded, margins)
    assert r["slope_per_decade"] > 0.5, r["slope_per_decade"]

    # a blind model: correctness independent of margin
    blind = [{"qid": f"q{i}", "gold": "superior",
              "pred": "superior" if (i * 7919) % 1000 < 700 else "inferior"}
             for i in range(400)]
    r2 = analyse(blind, margins)
    assert abs(r2["slope_per_decade"]) < 0.3, r2["slope_per_decade"]

    assert analyse([], {})["n"] == 0
    print(f"selftest OK — separates a geometry-reading model "
          f"(slope {r['slope_per_decade']:+.2f}) from a margin-blind one "
          f"(slope {r2['slope_per_decade']:+.2f})")


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        selftest()
    else:
        main()
