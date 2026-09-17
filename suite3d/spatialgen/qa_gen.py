"""
Turn a geometric scene graph into spatial QA whose answers are VERIFIABLE.

Report-derived spatial VQA extracts spatial facts from radiology TEXT. Here every
answer is computed from geometry and ships with the numeric margin that produced
it, so it can be re-checked.

Categories: laterality, longitudinal and anteroposterior relations, adjacency,
and extent.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, asdict, field

import numpy as np

from scene_graph import AXES, Structure, Relation

CATEGORIES = (
    "laterality",          # left / right / bilateral
    "longitudinal",        # superior / inferior
    "anteroposterior",     # anterior / posterior
    "mediolateral",        # central / peripheral
    "adjacency",           # adjacent / contained
    "extent",              # size, span, boundary
)

_AXIS_OF_CAT = {
    "laterality": "lateral",
    "longitudinal": "longitudinal",
    "anteroposterior": "anteroposterior",
}

_PHRASE = {
    "right_of": "to the patient's right of",
    "left_of": "to the patient's left of",
    "anterior_to": "anterior to",
    "posterior_to": "posterior to",
    "superior_to": "superior to",
    "inferior_to": "inferior to",
}

_OPPOSITE_WORD = {
    "right_of": "left", "left_of": "right",
    "anterior_to": "posterior", "posterior_to": "anterior",
    "superior_to": "inferior", "inferior_to": "superior",
}

_ANSWER_WORD = {
    "right_of": "right", "left_of": "left",
    "anterior_to": "anterior", "posterior_to": "posterior",
    "superior_to": "superior", "inferior_to": "inferior",
}


@dataclass
class QA:
    qid: str
    category: str
    question: str
    answer: str
    choices: list[str] = field(default_factory=list)
    # provenance is the whole point: an auditor can recompute `answer` from this
    provenance: dict = field(default_factory=dict)


def _pretty(name: str) -> str:
    return name.replace("_", " ")


def from_axis_relation(r: Relation, qid: str) -> QA | None:
    """Directional QA, e.g. 'Is the liver superior or inferior to the spleen?'"""
    if r.predicate not in _ANSWER_WORD:
        return None
    cat = next((c for c, a in _AXIS_OF_CAT.items() if a == r.axis), None)
    if cat is None:
        return None
    ans = _ANSWER_WORD[r.predicate]
    opp = _OPPOSITE_WORD[r.predicate]
    q = (f"In this CT volume, is the {_pretty(r.subject)} located {ans} or "
         f"{opp} to the {_pretty(r.object)}? Answer with one word.")
    return QA(
        qid=qid, category=cat, question=q, answer=ans,
        choices=sorted([ans, opp]),
        provenance={"relation": asdict(r), "rule": f"bbox separation on {r.axis}",
                    "margin_mm": r.margin_mm},
    )


def from_containment(r: Relation, qid: str) -> QA:
    q = (f"Is the {_pretty(r.subject)} located within the "
         f"{_pretty(r.object)}? Answer yes or no.")
    return QA(qid=qid, category="adjacency", question=q, answer="yes",
              choices=["no", "yes"],
              provenance={"relation": asdict(r), "rule": "voxel containment"})


def from_adjacency(r: Relation, qid: str, positive: bool = True) -> QA:
    q = (f"Does the {_pretty(r.subject)} directly contact or lie immediately "
         f"adjacent to the {_pretty(r.object)}? Answer yes or no.")
    return QA(qid=qid, category="adjacency", question=q,
              answer="yes" if positive else "no", choices=["no", "yes"],
              provenance={"relation": asdict(r), "rule": "surface distance"})


def from_laterality_absolute(
    s: Structure, qid: str, midline_x: float, tol_mm: float = 10.0
) -> QA | None:
    """Absolute laterality w.r.t. the body midline (not relative to another organ).

    Structures straddling the midline are labelled 'bilateral' rather than being
    forced to a side -- the same refuse-when-undecidable discipline as elsewhere.
    """
    lo, hi = s.extent(0)
    if lo > midline_x + tol_mm:
        ans = "right"
    elif hi < midline_x - tol_mm:
        ans = "left"
    elif lo < midline_x - tol_mm and hi > midline_x + tol_mm:
        ans = "bilateral"
    else:
        return None
    q = (f"Is the {_pretty(s.name)} located in the patient's left hemithorax/"
         f"hemiabdomen, the right, or is it bilateral/midline? Answer with one word.")
    return QA(qid=qid, category="laterality", question=q, answer=ans,
              choices=["bilateral", "left", "right"],
              provenance={"structure": s.name, "x_extent": [lo, hi],
                          "midline_x": midline_x, "rule": "midline comparison"})


def from_extent(s: Structure, qid: str, axis: int = 2) -> QA:
    """Quantitative extent along one axis, in millimetres."""
    lo, hi = s.extent(axis)
    span = round(hi - lo, 1)
    axis_name = {0: "left-right", 1: "anterior-posterior",
                 2: "cranio-caudal"}[axis]
    q = (f"What is the approximate {axis_name} extent of the "
         f"{_pretty(s.name)} in millimetres?")
    return QA(qid=qid, category="extent", question=q, answer=f"{span}",
              provenance={"structure": s.name, "axis": axis_name,
                          "bbox_lo": lo, "bbox_hi": hi,
                          "rule": "bbox span in RAS mm"})


def _side_of(name: str) -> str | None:
    """Detect the laterality tag in a structure name.

    TotalSegmentator names are SUFFIXED ('kidney_right', 'lung_upper_lobe_left')
    while many other atlases PREFIX ('right_lung'). Handle both — matching only
    one silently drops every paired structure and collapses the midline estimate
    to a plain centroid mean, which quietly corrupts all laterality labels.
    """
    parts = name.lower().split("_")
    if not parts:
        return None
    if parts[-1] in ("left", "right"):
        return parts[-1]
    if parts[0] in ("left", "right"):
        return parts[0]
    return None


def estimate_midline_x(structures: dict[str, Structure]) -> float:
    """Body midline in RAS x (mm).

    Preferred estimator: average the centroids of matched left/right pairs of
    the SAME organ (e.g. kidney_left + kidney_right), which cancels out
    asymmetric organ placement. Falls back to all lateralised structures, then
    to the global centroid.
    """
    pairs: dict[str, dict[str, float]] = {}
    lateralised: list[float] = []

    for name, s in structures.items():
        side = _side_of(name)
        if side is None:
            continue
        lateralised.append(s.centroid_ras[0])
        stem = "_".join(p for p in name.lower().split("_") if p != side)
        pairs.setdefault(stem, {})[side] = s.centroid_ras[0]

    matched = [(v["left"] + v["right"]) / 2.0
               for v in pairs.values() if "left" in v and "right" in v]
    if matched:
        return float(np.mean(matched))
    if len(lateralised) >= 2:
        return float(np.mean(lateralised))
    return float(np.mean([s.centroid_ras[0] for s in structures.values()]))


def generate(
    structures: dict[str, Structure],
    relations: list[Relation],
    volume_id: str,
    max_per_category: int = 40,
    seed: int = 0,
) -> list[QA]:
    """Build a balanced, deduplicated QA set for one volume."""
    rng = random.Random(seed)
    buckets: dict[str, list[QA]] = {c: [] for c in CATEGORIES}
    n = 0

    for r in relations:
        qid = f"{volume_id}_{n:05d}"
        qa = None
        if r.predicate in _ANSWER_WORD:
            qa = from_axis_relation(r, qid)
        elif r.predicate == "contained_in":
            qa = from_containment(r, qid)
        elif r.predicate == "adjacent_to":
            qa = from_adjacency(r, qid)
        if qa is not None:
            buckets[qa.category].append(qa)
            n += 1

    midline = estimate_midline_x(structures)
    for s in structures.values():
        qa = from_laterality_absolute(s, f"{volume_id}_{n:05d}", midline)
        if qa is not None:
            buckets["laterality"].append(qa)
            n += 1
        buckets["extent"].append(from_extent(s, f"{volume_id}_{n:05d}"))
        n += 1

    out: list[QA] = []
    for cat, items in buckets.items():
        rng.shuffle(items)
        out.extend(balance_labels(items, rng)[:max_per_category])
    return out


def balance_labels(items: list[QA], rng: random.Random) -> list[QA]:
    """Equalise gold-label counts within each choice pair.

    Without this the trivial "always answer the majority label" baseline sits
    above 50% and a blind model matches it for free -- measured at 58.6% on an
    unbalanced spleen shard, which makes an at-chance result unreadable. After
    balancing, chance is exactly 50% and any excess is real signal.

    Downsamples the majority label rather than duplicating the minority, so no
    question appears twice.
    """
    from collections import defaultdict

    groups: dict[tuple, dict[str, list[QA]]] = defaultdict(lambda: defaultdict(list))
    ungrouped: list[QA] = []
    for qa in items:
        if qa.choices and len(qa.choices) == 2:
            groups[tuple(sorted(qa.choices))][qa.answer].append(qa)
        else:
            ungrouped.append(qa)

    out: list[QA] = list(ungrouped)
    for _, by_label in groups.items():
        if len(by_label) < 2:
            continue                       # only one label present; nothing to balance
        k = min(len(v) for v in by_label.values())
        for label, qas in by_label.items():
            rng.shuffle(qas)
            out.extend(qas[:k])
    rng.shuffle(out)
    return out


def hard_negatives(qas: list[QA]) -> list[QA]:
    """Flip each directional QA into its inverse form.

    Pairing (A rel B) with (B inv-rel A) is what makes the antisymmetry axiom
    measurable at eval time: a language-prior model tends to answer both with
    the same word, which is geometrically impossible.
    """
    out = []
    for qa in qas:
        rel = qa.provenance.get("relation")
        if not rel or rel.get("predicate") not in _ANSWER_WORD:
            continue
        inv_pred = {"right_of": "left_of", "left_of": "right_of",
                    "anterior_to": "posterior_to", "posterior_to": "anterior_to",
                    "superior_to": "inferior_to", "inferior_to": "superior_to"}[
            rel["predicate"]]
        ans, opp = _ANSWER_WORD[inv_pred], _OPPOSITE_WORD[inv_pred]
        q = (f"In this CT volume, is the {_pretty(rel['object'])} located {ans} "
             f"or {opp} to the {_pretty(rel['subject'])}? Answer with one word.")

        # The provenance must describe THIS question, not the one it was derived
        # from. Copying the forward relation verbatim while inverting the answer
        # breaks the core contract of this dataset -- that any auditor can
        # recompute the answer from the provenance -- and makes exactly half of
        # the directional QA look wrong under an independent check.
        inv_rel = dict(rel)
        inv_rel["subject"], inv_rel["object"] = rel["object"], rel["subject"]
        inv_rel["predicate"] = inv_pred
        if rel.get("margin_mm") is not None:
            inv_rel["margin_mm"] = -rel["margin_mm"]
        ev = dict(rel.get("evidence") or {})
        if "a_extent" in ev and "b_extent" in ev:
            ev["a_extent"], ev["b_extent"] = ev["b_extent"], ev["a_extent"]
        inv_rel["evidence"] = ev

        inv_prov = {**qa.provenance, "relation": inv_rel,
                    "derived": "inverse_of", "source_qid": qa.qid}
        if inv_prov.get("margin_mm") is not None:
            inv_prov["margin_mm"] = -qa.provenance["margin_mm"]

        out.append(QA(qid=qa.qid + "_inv", category=qa.category, question=q,
                      answer=ans, choices=sorted([ans, opp]),
                      provenance=inv_prov))
    return out


def is_prior_answerable(qa: QA | dict) -> str | None:
    """Return a reason if this QA can be answered without looking at the volume.

    Found empirically: a blind text-only LLM (Qwen3-14B, no image) scored 66.0%
    on our generated QA against a 50% chance level, and 81.9% on the laterality
    subset. The cause is that many pairs have an answer fixed by naming or by
    canonical anatomy -- "is the kidney_left to the left or right of the
    kidney_right" is decided by the words alone. Such items are not spatial
    supervision; they teach the model to read structure names.

    By contrast the blind model scored 23.5% on adjacency, which is genuinely
    patient-specific. That contrast is the signal this filter chases.
    """
    d = qa if isinstance(qa, dict) else asdict(qa)
    prov = d.get("provenance") or {}
    rel = prov.get("relation")
    if not rel:
        return None
    a, b = rel.get("subject", ""), rel.get("object", "")
    cat = d.get("category")

    if cat == "laterality":
        sa, sb = _side_of(a), _side_of(b)
        if sa and sb and sa != sb:
            return "laterality decided by structure names"
        # a lateralised structure vs a midline one is also fixed by naming
        if (sa and not sb) or (sb and not sa):
            return "laterality of a named side vs an unlateralised structure"

    # bilateral counterparts of the same organ: every other axis is a coin flip
    # in principle but in practice near-constant, so keep them only for adjacency
    if _side_of(a) and _side_of(b):
        stem_a = "_".join(p for p in a.split("_") if p not in ("left", "right"))
        stem_b = "_".join(p for p in b.split("_") if p not in ("left", "right"))
        if stem_a == stem_b and cat in ("longitudinal", "anteroposterior"):
            return "mirror-image counterparts have a near-fixed relation"
    return None


def filter_prior_answerable(qas: list[QA]) -> tuple[list[QA], dict[str, int]]:
    """Drop QA answerable from naming/anatomical priors; report why."""
    kept, reasons = [], {}
    for qa in qas:
        why = is_prior_answerable(qa)
        if why:
            reasons[why] = reasons.get(why, 0) + 1
        else:
            kept.append(qa)
    return kept, reasons


def constant_relation_keys(
    per_volume_relations: list[list[dict]], min_volumes: int = 5
) -> set[tuple[str, str, str]]:
    """(subject, object, axis) triples whose answer never varies across volumes.

    The principled version of the filter above: a relation that is identical in
    every scan carries no information about any particular scan, whatever the
    reason. Requires several volumes to be meaningful, hence min_volumes.
    """
    from collections import defaultdict

    seen: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    counts: dict[tuple[str, str, str], int] = defaultdict(int)
    # Every structure pair observed in a volume, so that a pair which is
    # adjacent in some scans and not in others counts as VARYING. Judging
    # adjacency only from the volumes where it fired would call every such pair
    # constant, which is backwards: presence/absence is exactly the signal.
    pair_seen: dict[tuple[str, str], int] = defaultdict(int)
    pair_adj: dict[tuple[str, str], int] = defaultdict(int)

    for rels in per_volume_relations:
        structures = {r["subject"] for r in rels} | {r["object"] for r in rels}
        adj_here = set()
        for r in rels:
            if r.get("axis"):
                key = (r["subject"], r["object"], r["axis"])
                seen[key].add(r["predicate"])
                counts[key] += 1
            elif r["predicate"] == "adjacent_to":
                adj_here.add(tuple(sorted((r["subject"], r["object"]))))
        for a in sorted(structures):
            for b in sorted(structures):
                if a < b:
                    pair_seen[(a, b)] += 1
                    if (a, b) in adj_here:
                        pair_adj[(a, b)] += 1

    const = {k for k, preds in seen.items()
             if len(preds) == 1 and counts[k] >= min_volumes}

    # an adjacency that is always present, or never present, in every volume
    # where both structures exist tells nothing about a particular volume
    for pair, n in pair_seen.items():
        if n < min_volumes:
            continue
        hits = pair_adj.get(pair, 0)
        if hits == 0 or hits == n:
            const.add((pair[0], pair[1], "adjacency"))
    return const


_POS_WORD = {"laterality": "right", "longitudinal": "superior",
             "anteroposterior": "anterior"}
_NEG_WORD = {"laterality": "left", "longitudinal": "inferior",
             "anteroposterior": "posterior"}


def validate_provenance(qas: list[QA] | list[dict]) -> list[dict]:
    """Recompute every directional answer from its provenance; return mismatches.

    This is the dataset's core invariant: an independent auditor holding only the
    provenance must arrive at the stated answer. Run it on every generated shard
    -- a silent violation here corrupts the ground truth without any downstream
    signal, which is precisely the failure mode this project accuses
    report-derived benchmarks of.
    """
    bad = []
    for qa in qas:
        d = qa if isinstance(qa, dict) else asdict(qa)
        prov = d.get("provenance") or {}
        rel = prov.get("relation")
        cat = d.get("category")
        if not rel or cat not in _POS_WORD:
            continue
        if prov.get("rule") == "midline comparison":
            continue
        margin = rel.get("margin_mm")
        if margin is None:
            continue
        expect = _POS_WORD[cat] if margin > 0 else _NEG_WORD[cat]
        if d["answer"] != expect:
            bad.append({"qid": d["qid"], "answer": d["answer"],
                        "expected_from_provenance": expect,
                        "margin_mm": margin, "category": cat})
    return bad


def selftest() -> None:
    from scene_graph import build_structures, build_scene_graph

    # Scale matters here: structures must clear the midline by more than
    # from_laterality_absolute's tol_mm, otherwise they are correctly refused as
    # undecidable. Use a body-sized synthetic volume (200mm across).
    seg = np.zeros((200, 100, 100), dtype=np.int16)
    affine = np.eye(4)
    seg[130:180, 30:60, 30:60] = 1     # well right of midline (x~155)
    seg[20:70, 30:60, 30:60] = 2       # well left of midline  (x~45)
    seg[80:120, 30:60, 70:95] = 3      # straddles midline, superior
    names = {1: "right_lung", 2: "left_lung", 3: "top_organ"}

    st = build_structures(seg, affine, names)
    rels = build_scene_graph(st, seg, affine, do_adjacency=False)
    qas = generate(st, rels, "synth001")
    assert qas, "no QA generated"

    # every answer must be recomputable from its provenance
    for qa in qas:
        assert qa.answer, qa
        assert qa.provenance, qa

    lat = [q for q in qas if q.category == "laterality"]
    assert lat, "no laterality QA"
    # right_lung sits at high x -> patient right
    abs_lat = [q for q in lat if q.provenance.get("rule") == "midline comparison"]
    by_struct = {q.provenance["structure"]: q.answer for q in abs_lat}
    assert by_struct.get("right_lung") == "right", by_struct
    assert by_struct.get("left_lung") == "left", by_struct

    inv = hard_negatives(qas)
    assert inv, "no inverse QA"
    for q in inv:
        assert q.qid.endswith("_inv")

    # the invariant that makes this dataset auditable, checked on both halves
    bad = validate_provenance(qas + inv)
    assert not bad, f"provenance does not reproduce answers: {bad[:3]}"

    # and specifically: an inverse QA must carry the SWAPPED, NEGATED relation
    sample = inv[0]
    src = next(q for q in qas if q.qid == sample.provenance["source_qid"])
    sr, ir = src.provenance["relation"], sample.provenance["relation"]
    assert ir["subject"] == sr["object"] and ir["object"] == sr["subject"], ir
    assert ir["margin_mm"] == -sr["margin_mm"], (ir["margin_mm"], sr["margin_mm"])

    cats = {q.category for q in qas}
    print(f"selftest OK — {len(qas)} QA over {sorted(cats)}, "
          f"{len(inv)} inverse pairs for the antisymmetry probe; "
          f"provenance reproduces all {len(validate_provenance(qas+inv)) == 0 and 'answers' or '?'}")


if __name__ == "__main__":
    selftest()
