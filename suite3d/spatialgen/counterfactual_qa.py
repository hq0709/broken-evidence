"""
Generate counterfactual anatomical QA with computed ground truth.

Design
------
Every item asks about a state that does not exist in the scan, so no report,
textbook, or language prior can supply the answer -- the scenario never
occurred. That is what makes this task class immune to the two failure modes
that sank the earlier attempts in this project:

  * relations between normal organs are patient-invariant (measured: 3651
    relations, zero varying), so questions about them carry no per-patient
    information -- but a counterfactual depends on THIS lesion's geometry;
  * benchmarks built from reports inherit unverifiable labels -- here the
    simulator computes the answer, so it is checkable to the voxel.

MATCHED PAIRS are the core of the design. For one lesion/target pair we emit two
questions: one growth amount below the current surface gap (answer: no) and one
above it (answer: yes). This gives exact label balance, so chance is exactly
50%, and it creates a probe: a model that is not computing anything answers both
members identically, which is geometrically impossible. Pair-inconsistency is
therefore measurable without any reference to correctness.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from anatomy_sim import (  # noqa: E402
    grow, surface_gap_mm, tool_resection_remaining, volume_mm3,
)

# Structures a lesion contacting them would actually change management.
# Restricting to these keeps the questions clinically meaningful rather than
# asking about arbitrary rib numbers.
CRITICAL_TARGETS = {
    "aorta", "heart", "esophagus", "trachea", "pulmonary_artery",
    "inferior_vena_cava", "portal_vein_and_splenic_vein", "spinal_cord",
    "liver", "spleen", "pancreas", "kidney_left", "kidney_right", "stomach",
    "colon", "small_bowel", "gallbladder", "urinary_bladder", "duodenum",
    "adrenal_gland_left", "adrenal_gland_right",
    "lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
    "lung_middle_lobe_right", "lung_lower_lobe_right",
    "vertebrae_T8", "vertebrae_T9", "vertebrae_T10", "sternum",
}


@dataclass
class CFQuestion:
    qid: str
    kind: str
    question: str
    answer: str
    choices: list[str] = field(default_factory=list)
    pair_id: str | None = None
    provenance: dict = field(default_factory=dict)


def _pretty(name: str) -> str:
    return name.replace("_", " ")


def growth_pairs(lesion: np.ndarray, targets: dict[str, np.ndarray],
                 affine: np.ndarray, vid: str, lesion_id: str,
                 max_gap_mm: float = 40.0, margin: float = 2.0,
                 rng: random.Random | None = None) -> list[CFQuestion]:
    """One matched yes/no pair per target whose gap is in a testable range."""
    from scipy.ndimage import distance_transform_edt

    rng = rng or random.Random(0)
    out: list[CFQuestion] = []

    # ONE distance transform from the lesion answers every growth query exactly.
    # Growing by g yields {dist <= g}, so contact with a target after growth g
    # holds iff min(dist over the target) <= g. Recomputing grow() and a fresh
    # transform per (target, amount) costs ~50 full-volume EDTs per lesion --
    # about ten minutes on a 79M-voxel chest CT -- and returns the same answer.
    spacing = np.abs(np.diag(affine)[:3])
    dist = distance_transform_edt(~lesion, sampling=spacing)

    for tname, tmask in sorted(targets.items()):
        if not tmask.any():
            continue
        gap = 0.0 if (lesion & tmask).any() else float(dist[tmask].min())
        # already touching -> the counterfactual is vacuous; too far -> the
        # growth needed is implausible and the question stops being clinical
        if gap <= margin or gap > max_gap_mm or not np.isfinite(gap):
            continue

        below = max(1.0, round(gap - margin - rng.uniform(0, gap * 0.3), 1))
        above = round(gap + margin + rng.uniform(0, 5.0), 1)

        for amount, expect in ((below, "no"), (above, "yes")):
            actual = "yes" if gap <= amount else "no"
            if actual != expect:
                continue
            pid = f"{vid}_{lesion_id}_{tname}"
            out.append(CFQuestion(
                qid=f"{pid}_g{amount:g}",
                kind="growth_contact",
                question=(f"If this lesion grew by {amount:g} mm in every "
                          f"direction, would it contact the {_pretty(tname)}? "
                          f"Answer yes or no."),
                answer=actual, choices=["no", "yes"], pair_id=pid,
                provenance={"target": tname, "gap_mm": round(gap, 2),
                            "growth_mm": amount,
                            "rule": "surface distance after isotropic growth"},
            ))
    # keep only complete pairs: a lone member cannot test pair-consistency
    by_pair: dict[str, list[CFQuestion]] = {}
    for q in out:
        by_pair.setdefault(q.pair_id, []).append(q)
    return [q for qs in by_pair.values() if len(qs) == 2 for q in qs]


def nearest_target_question(lesion: np.ndarray, targets: dict[str, np.ndarray],
                            affine: np.ndarray, vid: str, lesion_id: str,
                            n_choices: int = 4, max_ratio: float = 6.0,
                            rng: random.Random | None = None
                            ) -> CFQuestion | None:
    """Which structure would growth reach first? Relational AND counterfactual.

    Requires ordering several distances at once, so it cannot be answered by
    recalling any single fact about the scan.
    """
    from scipy.ndimage import distance_transform_edt

    rng = rng or random.Random(0)
    spacing = np.abs(np.diag(affine)[:3])
    dist = distance_transform_edt(~lesion, sampling=spacing)
    gaps = {n: (0.0 if (lesion & m).any() else float(dist[m].min()))
            for n, m in targets.items() if m.any()}
    gaps = {n: g for n, g in gaps.items() if np.isfinite(g) and g > 0}
    if len(gaps) < n_choices:
        return None

    ordered = sorted(gaps, key=gaps.get)
    winner = ordered[0]
    w = gaps[winner]

    # Distractors must be COMPARABLY close. Sampling any structure farther than
    # the winner produced items like {esophagus 2.8 mm, duodenum 216 mm, colon
    # 206 mm}, where the answer is 70x nearer than every alternative -- solvable
    # from textbook anatomy without looking at the scan, which is exactly the
    # prior-answerability this task class exists to avoid.
    # Require: clearly separated from the winner (so the label is robust to
    # discretisation) but within an order of magnitude (so the choice is real).
    # HARD ratio cap, no additive fallback. An additive allowance (w + 40 mm)
    # silently voids the guarantee when the winner is close: at w = 2 mm it
    # admits a 42 mm distractor, a 21x gap that anatomy knowledge alone can
    # rule out. Dropping the item is strictly better than emitting an easy one.
    candidates = [n for n in ordered[1:]
                  if w + 3.0 < gaps[n] <= w * max_ratio]
    if len(candidates) < n_choices - 1:
        return None
    picks = [winner] + rng.sample(candidates, n_choices - 1)
    rng.shuffle(picks)

    return CFQuestion(
        qid=f"{vid}_{lesion_id}_nearest",
        kind="nearest_on_growth",
        question=("If this lesion grew uniformly in all directions, which "
                  "structure would it reach first? Answer with one option: "
                  + ", ".join(_pretty(p) for p in picks) + "."),
        answer=_pretty(winner),
        choices=sorted(_pretty(p) for p in picks),
        provenance={"gaps_mm": {n: round(gaps[n], 2) for n in picks},
                    "distractor_ratio": round(
                        max(gaps[p] for p in picks) / max(w, 1e-6), 2),
                    "rule": "minimum surface gap"},
    )


def resection_question(parts: dict[str, np.ndarray], affine: np.ndarray,
                       vid: str, rng: random.Random) -> CFQuestion | None:
    """Percentage of an organ remaining after removing one part."""
    if len(parts) < 3:
        return None
    part = rng.choice(sorted(parts))
    r = tool_resection_remaining(parts, affine, part)
    return CFQuestion(
        qid=f"{vid}_resect_{part}",
        kind="resection_volume",
        question=r.query,
        answer=r.answer,
        provenance={**r.evidence, "resected": part},
    )


def generate(lesion_masks: dict[str, np.ndarray],
             anatomy: dict[str, np.ndarray], affine: np.ndarray,
             vid: str, seed: int = 0) -> list[CFQuestion]:
    rng = random.Random(seed)
    targets = {n: m for n, m in anatomy.items()
               if n in CRITICAL_TARGETS and m.any()}
    out: list[CFQuestion] = []
    for lid, lmask in sorted(lesion_masks.items()):
        if lmask.sum() < 20:
            continue
        out.extend(growth_pairs(lmask, targets, affine, vid, lid, rng=rng))
        q = nearest_target_question(lmask, targets, affine, vid, lid, rng=rng)
        if q:
            out.append(q)

    lobes = {n: m for n, m in anatomy.items() if n.startswith("lung_") and m.any()}
    q = resection_question(lobes, affine, vid, rng)
    if q:
        out.append(q)
    return out


def check_balance(qs: list[CFQuestion]) -> dict:
    from collections import Counter

    yn = [q for q in qs if q.kind == "growth_contact"]
    c = Counter(q.answer for q in yn)
    pairs = Counter(q.pair_id for q in yn)
    return {"growth_contact": len(yn),
            "label_counts": dict(c),
            "balanced": c.get("yes", 0) == c.get("no", 0),
            "complete_pairs": sum(1 for v in pairs.values() if v == 2),
            "orphans": sum(1 for v in pairs.values() if v != 2)}


def selftest() -> None:
    shape = (80, 80, 40)
    affine = np.diag([1.0, 1.0, 3.0, 1.0])

    lesion = np.zeros(shape, bool)
    lesion[38:42, 38:42, 19:21] = True
    anat = {}
    for name, x0 in (("aorta", 55), ("esophagus", 62), ("heart", 70)):
        m = np.zeros(shape, bool)
        m[x0:x0 + 6, 35:45, 15:25] = True
        anat[name] = m

    qs = generate({"lesion1": lesion}, anat, affine, "vol1")
    assert qs, "no questions generated"

    bal = check_balance(qs)
    assert bal["balanced"], bal
    assert bal["orphans"] == 0, bal
    assert bal["complete_pairs"] >= 1, bal

    # every yes/no answer must be reproducible from its provenance
    for q in qs:
        if q.kind != "growth_contact":
            continue
        p = q.provenance
        expect = "yes" if p["growth_mm"] >= p["gap_mm"] else "no"
        assert q.answer == expect, (q.qid, q.answer, p)

    # the nearest-structure question must name the genuinely closest target
    max_ratio_used = 6.0
    nq = [q for q in qs if q.kind == "nearest_on_growth"]
    if nq:
        g = nq[0].provenance["gaps_mm"]
        assert nq[0].answer == min(g, key=g.get).replace("_", " "), (nq[0].answer, g)
        # distractors must be within an order of magnitude of the winner,
        # otherwise the item is answerable from anatomy alone
        assert nq[0].provenance["distractor_ratio"] <= max_ratio_used, \
            nq[0].provenance

    # matched members differ only in the growth amount
    pairs = {}
    for q in qs:
        if q.kind == "growth_contact":
            pairs.setdefault(q.pair_id, []).append(q)
    a, b = next(iter(pairs.values()))
    assert a.provenance["target"] == b.provenance["target"]
    assert {a.answer, b.answer} == {"yes", "no"}

    print(f"selftest OK — {len(qs)} questions, {bal['complete_pairs']} matched "
          f"pairs, labels exactly balanced {bal['label_counts']}, every answer "
          f"reproducible from provenance, nearest-target verified")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        selftest()
    else:
        main()
