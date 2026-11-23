"""exp_R2_4_poolsize.py - Reviewer R2.4: arm-C (nearest real benign flow) pool-size sensitivity, with
by-class coverage and distance. Re-runs the nearest-real baseline on the confirmatory cases at nested
donor-pool sizes {25k, 50k, 100k, 200k(full)} on the REBUILT models, and reports:
  - coverage (fraction of cases with a matching benign flow sharing the immutable features),
  - percentile (M4) and Gower (M5) median distance among covered cases,
  - the eligible-benign pool size actually available, and the per-case immutable-matched pool size,
  - all of the above broken down BY ATTACK CLASS.
Nested pools (smaller is a subset of larger) via one seeded permutation, so the curve is monotone in
information, not in sampling noise. Checkpoint: skips a (ds,det,K) whose npz already exists.
Usage: python exp_R2_4_poolsize.py <cic|unsw> <RF|XGB|HGB>
"""
import os, sys, time, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace'); sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
import expbase as EB
from generator import p_benign
from expcommon import metrics_from_array

ds, det = sys.argv[1], sys.argv[2]
POOL_SIZES = [25000, 50000, 100000, 200000]
OUT = os.path.join(EB.EXP, 'R2_4_poolsize', 'raw'); os.makedirs(OUT, exist_ok=True)
sp, feats, meta = EB.build_space(ds)
model = EB.load_model(ds, det)
X0, c0, idx = EB.conf_cases(ds, feats)

full = len(sp.donors)
perm = np.random.default_rng(EB.SEED).permutation(full)   # one nested permutation for all K


def nearest_real_subpool(sp, model, X0, local_idx):
    """nearest_real restricted to the donor rows in local_idx (indices into sp.donors)."""
    pool = sp.donors[local_idx]
    orig_rows = sp.ben_pool_idx[local_idx]
    keep = p_benign(model, pool) > 0.5
    pool = pool[keep]; orig_rows = orig_rows[keep]
    Pp = sp.pct(pool).astype(np.float32); P0 = sp.pct(X0).astype(np.float32)
    cat = sp.cat; N = len(X0)
    cf = X0.copy(); found = np.zeros(N, bool); pool_row = -np.ones(N, np.int64)
    matched = np.zeros(N, np.int64)
    imm_idx = np.flatnonzero(sp.immutable)
    for i in range(N):
        el = np.ones(len(pool), bool)
        for j in imm_idx:
            el &= pool[:, j] == X0[i, j]
        matched[i] = int(el.sum())
        if not el.any():
            continue
        ii = np.flatnonzero(el)
        d = np.abs(Pp[ii] - P0[i]); d[:, cat] = d[:, cat] > 0
        b = ii[np.argmin(d.sum(1))]
        cf[i] = pool[b]; found[i] = True; pool_row[i] = orig_rows[b]
    return dict(cf=cf, returned=found, p_benign=p_benign(model, cf), pool_size=int(len(pool)),
                train_row=pool_row, matched_pool=matched)


summary = {'ds': ds, 'det': det, 'full_donor_pool': full, 'sizes': {}}
for K in POOL_SIZES:
    K = min(K, full)
    fn = os.path.join(OUT, f'{ds}_{det}_C_pool{K}.npz')
    if os.path.exists(fn):
        print(f'skip pool{K} (exists)', flush=True)
        z = np.load(fn, allow_pickle=True)
    else:
        t = time.time()
        r = nearest_real_subpool(sp, model, X0, perm[:K])
        EB.save_arm(fn, sp, feats, 'C', 0, X0, c0, idx, r, extra={'K_requested': K, 'matched_pool': r['matched_pool']})
        print(f'{ds} {det} pool{K}: coverage {r["returned"].mean():.3f} elig_pool {r["pool_size"]} '
              f'[{time.time()-t:.0f}s]', flush=True)
        z = np.load(fn, allow_pickle=True)
    m = metrics_from_array(z, ds, is_C=True)
    cov = z['returned']; dpct = z['d_pct']; dgow = z['d_gower']
    by_class = {}
    for cls in sorted(set(c0.tolist())):
        sel = (c0 == cls); covsel = sel & cov
        by_class[str(cls)] = {
            'n': int(sel.sum()), 'coverage': float(cov[sel].mean()) if sel.any() else None,
            'M4_pct_med': float(np.median(dpct[covsel])) if covsel.any() else None,
            'M5_gower_med': float(np.median(dgow[covsel])) if covsel.any() else None,
            'matched_pool_med': int(np.median(z['matched_pool'][sel])) if sel.any() else None}
    summary['sizes'][str(K)] = {
        'elig_benign_pool': int(z['pool_size']), 'coverage': float(cov.mean()),
        'M1': m['M1'], 'M3': m['M3'],
        'M4_pct_med': float(np.median(dpct[cov])) if cov.any() else None,
        'M5_gower_med': float(np.median(dgow[cov])) if cov.any() else None,
        'by_class': by_class}

os.makedirs(os.path.join(EB.EXP, 'R2_4_poolsize'), exist_ok=True)
json.dump(summary, open(os.path.join(EB.EXP, 'R2_4_poolsize', f'{ds}_{det}_summary.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print(f'R2.4 {ds}_{det} done: ' + ' '.join(f'K{k}:cov={v["coverage"]:.3f}' for k, v in summary['sizes'].items()), flush=True)
