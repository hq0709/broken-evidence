"""Four-organ, two-model summary: image contribution, pair consistency, intervention."""
import json, glob, random
from collections import defaultdict, Counter
import os as _os
R = _os.environ.get("VOLUMETRIC_ROOT",
                    _os.path.dirname(_os.path.abspath(__file__)))
R = R if R.endswith("/") else R + "/"
ORGANS = ['Task06_Lung', 'Task10_Colon', 'Task07_Pancreas', 'Task03_Liver']

def vid(q):
    p = q.split('_')
    for i, x in enumerate(p):
        if x.startswith('lesion') or x == 'resect':
            return '_'.join(p[:i])
    return '_'.join(p[:2])

def load(tag, model):
    S = {json.loads(l)['qid']: json.loads(l) for l in open(f'{R}sc_{tag}_{model}_sighted.jsonl')}
    B = {json.loads(l)['qid']: json.loads(l) for l in open(f'{R}sc_{tag}_{model}_blind.jsonl')}
    qa = {json.loads(l)['qid']: json.loads(l)
          for f in glob.glob(f'{R}cfqa_{tag}/qa/*.jsonl') for l in open(f)}
    return S, B, qa

def amp_cv(S, B, folds=10, alphas=(0.5,1,2,3,5,8,12), seed=0):
    byv = defaultdict(list)
    for q in S:
        if q in B: byv[vid(q)].append(q)
    vols = sorted(byv); rng = random.Random(seed); rng.shuffle(vols)
    chunks = [vols[i::folds] for i in range(folds)]
    def sc(qs, a):
        ok = n = 0
        for q in qs:
            s, b = S[q], B[q]
            ks = [k for k in s['logprobs'] if k in b['logprobs']]
            if len(ks) < 2: continue
            d = {k: b['logprobs'][k] + a*(s['logprobs'][k]-b['logprobs'][k]) for k in ks}
            ok += max(d, key=d.get) == s['gold']; n += 1
        return ok, n
    to = tb = tn = 0; pick = []
    for i, held in enumerate(chunks):
        tr = [q for j, c in enumerate(chunks) if j != i for v in c for q in byv[v]]
        te = [q for v in held for q in byv[v]]
        if not tr or not te: continue
        best = max(alphas, key=lambda a: (lambda o, n: o/max(n,1))(*sc(tr, a)))
        pick.append(best); o, n = sc(te, best); to += o; tn += n; tb += sc(te, 1.0)[0]
    return tb, to, tn, Counter(pick)

def boot_cv(S, B, folds=5, alphas=(0.5,1,2,3,5,8,12), n=500, seed=0):
    """Patient-level bootstrap of the WHOLE cross-validated alpha procedure.

    The interval must describe the quantity actually reported. Resampling with
    a single fixed alpha understates the spread, because choosing alpha from
    data is itself a source of variance -- and it can put the point estimate
    outside its own interval.
    """
    byv = defaultdict(list)
    for q in S:
        if q in B: byv[vid(q)].append(q)
    vols = sorted(byv); rng = random.Random(seed)
    def sc(qs, a):
        ok = 0
        for q in qs:
            s, b = S[q], B[q]
            ks = [k for k in s['logprobs'] if k in b['logprobs']]
            if len(ks) < 2: continue
            d = {k: b['logprobs'][k] + a*(s['logprobs'][k]-b['logprobs'][k]) for k in ks}
            ok += max(d, key=d.get) == s['gold']
        return ok
    def gain(vs):
        chunks = [vs[i::folds] for i in range(folds)]
        to = tb = tn = 0
        for i, held in enumerate(chunks):
            tr = [q for j, c in enumerate(chunks) if j != i for v in c for q in byv[v]]
            te = [q for v in held for q in byv[v]]
            if not tr or not te: continue
            best = max(alphas, key=lambda a: sc(tr, a))
            to += sc(te, best); tb += sc(te, 1.0); tn += len(te)
        return 100.0*(to-tb)/tn if tn else 0.0
    ds = []
    for _ in range(n):
        samp = [rng.choice(vols) for _ in vols]; rng.shuffle(samp)
        ds.append(gain(samp))
    ds.sort()
    return ds[int(.025*n)], ds[int(.975*n)], sum(1 for d in ds if d <= 0)/n


print(f"{'organ':16}{'model':6}{'vols':>5}{'n':>7}{'sighted':>8}{'blind':>8}{'gain':>8}"
      f"{'pair-viol':>10}{'α=1':>8}{'CV-α':>8}{'interv':>8}{'95%CI':>16}{'P(Δ≤0)':>9}")
rows = []
for T in ORGANS:
    for m in ['m3d', 'qwen']:
        try: S, B, qa = load(T, m)
        except FileNotFoundError: continue
        sa = sum(S[q]['prediction'] == S[q]['gold'] for q in S)/len(S)
        ba = sum(B[q]['prediction'] == B[q]['gold'] for q in B)/len(B)
        g = defaultdict(list)
        for q, r in S.items():
            pid = qa.get(q, {}).get('pair_id')
            if pid: g[pid].append(r['prediction'])
        c = {k: v for k, v in g.items() if len(v) == 2}
        viol = sum(1 for v in c.values() if v[0] == v[1])/max(len(c), 1)
        tb, to, tn, pk = amp_cv(S, B)
        lo, hi, p0 = boot_cv(S, B)
        nv = len({vid(q) for q in S})
        print(f"{T:16}{m:6}{nv:5d}{len(S):7d}{sa:8.1%}{ba:8.1%}{100*(sa-ba):+8.1f}{viol:10.1%}"
              f"{100*tb/tn:8.1f}{100*to/tn:8.1f}{100*(to-tb)/tn:+8.1f}"
              f"  [{lo:+5.1f},{hi:+5.1f}]{p0:9.3f}")
        rows.append((T, m, nv, len(S), sa, ba, viol, tb/tn, to/tn))

print("\n=== matched-pair consistency as a label-free signal ===")
for T in ORGANS:
    for m in ['m3d', 'qwen']:
        try: S, B, qa = load(T, m)
        except FileNotFoundError: continue
        g = defaultdict(list)
        for q, r in S.items():
            pid = qa.get(q, {}).get('pair_id')
            if pid: g[pid].append((q, r))
        cons, inc = [], []
        for pid, items in g.items():
            if len(items) != 2: continue
            preds = [r['prediction'] for _, r in items]
            ok = [r['prediction'] == r['gold'] for _, r in items]
            (inc if preds[0] == preds[1] else cons).extend(ok)
        if cons:
            print(f"  {T:16}{m:6} consistent {100*sum(cons)/len(cons):5.1f}% (n={len(cons):5d})  "
                  f"violating {100*sum(inc)/len(inc):5.1f}% (n={len(inc):5d})  "
                  f"coverage {100*len(cons)/(len(cons)+len(inc)):4.1f}%")
        else:
            print(f"  {T:16}{m:6} no consistent pair (violation 100%)")
