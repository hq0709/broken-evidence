"""Cross-model summary on the common 2,262-probe subset."""
import glob
import json
import os
import random
from collections import Counter, defaultdict

import os as _os
R = _os.environ.get("VOLUMETRIC_ROOT",
                    _os.path.dirname(_os.path.abspath(__file__)))
R = R if R.endswith("/") else R + "/"
# general-purpose (montage-fed) first, ascending in size, then the native
# volumetric systems -- the split that every result in the audit falls along
ORDER = ['smolvlm', 'qwen3b', 'qwen7b', 'internvl', 'qwen3vl', 'internvl14',
         'qwen32b', 'llavaov', 'idefics3', 'pixtral',
         'm3d', 'm3dllama', 'med3dvlm']
LABEL = {'smolvlm': 'SmolVLM2-2.2B', 'qwen3b': 'Qwen2.5-VL-3B',
         'qwen7b': 'Qwen2.5-VL-7B', 'internvl': 'InternVL3-8B',
         'qwen3vl': 'Qwen3-VL-8B', 'internvl14': 'InternVL3-14B',
         'qwen32b': 'Qwen2.5-VL-32B', 'llavaov': 'LLaVA-OneVision-7B',
         'idefics3': 'Idefics3-8B', 'pixtral': 'Pixtral-12B',
         'm3d': 'M3D-LaMed-Phi3-4B', 'm3dllama': 'M3D-LaMed-Llama2-7B',
         'med3dvlm': 'Med3DVLM-7B'}


def vol(qid):
    return '_'.join(qid.split('_')[:2])


def load(name, arm):
    p = f'{R}mm_{name}_{arm}.jsonl'
    if not os.path.exists(p):
        return None
    return {json.loads(l)['qid']: json.loads(l) for l in open(p)}


def amp_cv(S, B, folds=10, alphas=(0.5, 1, 2, 3, 5, 8, 12), seed=0):
    byv = defaultdict(list)
    for q in S:
        if q in B:
            byv[vol(q)].append(q)
    vols = sorted(byv)
    rng = random.Random(seed)
    rng.shuffle(vols)
    chunks = [vols[i::folds] for i in range(folds)]

    def sc(qs, a):
        ok = n = 0
        for q in qs:
            s, b = S[q], B[q]
            ks = [k for k in s['logprobs'] if k in b['logprobs']]
            if len(ks) < 2:
                continue
            d = {k: b['logprobs'][k] + a * (s['logprobs'][k] - b['logprobs'][k])
                 for k in ks}
            ok += max(d, key=d.get) == s['gold']
            n += 1
        return ok, n

    to = tb = tn = 0
    pick = []
    for i, held in enumerate(chunks):
        tr = [q for j, c in enumerate(chunks) if j != i for v in c for q in byv[v]]
        te = [q for v in held for q in byv[v]]
        if not tr or not te:
            continue
        best = max(alphas, key=lambda a: (lambda o, n: o / max(n, 1))(*sc(tr, a)))
        pick.append(best)
        o, n = sc(te, best)
        to += o
        tn += n
        tb += sc(te, 1.0)[0]
    return (tb / tn if tn else 0), (to / tn if tn else 0), tn, Counter(pick)


def boot_ci(S, B, folds=5, alphas=(0.5, 1, 2, 3, 5, 8, 12), n=500, seed=0):
    """Bootstrap the ENTIRE cross-validated procedure, not a fixed alpha.

    A point estimate from per-fold alpha selection and an interval from one
    fixed alpha are different estimators. Resampling patients and re-running
    the whole selection inside each resample makes the interval describe the
    quantity actually reported, and absorbs the extra variance that choosing
    alpha from data introduces.
    """
    byv = defaultdict(list)
    for q in S:
        if q in B:
            byv[vol(q)].append(q)
    vols = sorted(byv)
    rng = random.Random(seed)

    def sc(qs, a):
        ok = 0
        for q in qs:
            s, b = S[q], B[q]
            ks = [k for k in s['logprobs'] if k in b['logprobs']]
            if len(ks) < 2:
                continue
            d = {k: b['logprobs'][k] + a * (s['logprobs'][k] - b['logprobs'][k])
                 for k in ks}
            ok += max(d, key=d.get) == s['gold']
        return ok

    def cv_gain(vs):
        """Held-out gain of CV-selected alpha over alpha=1, on this patient set."""
        chunks = [vs[i::folds] for i in range(folds)]
        to = tb = tn = 0
        for i, held in enumerate(chunks):
            tr = [q for j, c in enumerate(chunks) if j != i for v in c
                  for q in byv[v]]
            te = [q for v in held for q in byv[v]]
            if not tr or not te:
                continue
            best = max(alphas, key=lambda a: sc(tr, a))
            to += sc(te, best)
            tb += sc(te, 1.0)
            tn += len(te)
        return (to - tb) / tn if tn else 0.0

    ds = []
    for _ in range(n):
        samp = [rng.choice(vols) for _ in vols]
        rng.shuffle(samp)
        ds.append(cv_gain(samp))
    ds.sort()
    return ds[int(.025 * n)], ds[int(.975 * n)]


print(f"{'model':18}{'n':>6}{'sighted':>8}{'blind':>8}{'gain':>8}"
      f"{'pair-viol':>10}{'CV-α':>8}{'interv':>8}{'95%CI':>18}")
for name in ORDER:
    S, B = load(name, 'sighted'), load(name, 'blind')
    if not S or not B:
        print(f"{LABEL.get(name, name):18}{'-- not run --':>30}")
        continue
    common = [q for q in S if q in B]
    sa = sum(S[q]['prediction'] == S[q]['gold'] for q in common) / len(common)
    ba = sum(B[q]['prediction'] == B[q]['gold'] for q in common) / len(common)
    g = defaultdict(list)
    for q in common:
        pid = S[q].get('pair_id')
        if pid:
            g[pid].append(S[q]['prediction'])
    c = {k: v for k, v in g.items() if len(v) == 2}
    viol = sum(1 for v in c.values() if v[0] == v[1]) / max(len(c), 1)
    base, cv, tn, pk = amp_cv(S, B)
    a = pk.most_common(1)[0][0] if pk else 1
    lo, hi = boot_ci(S, B)
    print(f"{LABEL.get(name, name):18}{len(common):6d}{sa:8.1%}{ba:8.1%}"
          f"{100*(sa-ba):+8.1f}{viol:10.1%}{a:8}{100*(cv-base):+8.1f}"
          f"  [{100*lo:+5.1f},{100*hi:+5.1f}]")

print("\nchance (exact by construction) 50.0%   common subset 2262 probes (1131 yes / 1131 no)")
