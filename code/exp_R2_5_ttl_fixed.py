"""exp_R2_5_ttl_fixed.py - Reviewer R2.5: TTL-fixed regeneration (UNSW only). Re-run the genetic arms
A_typed and B with sttl and dttl added to the immutable set, on the REBUILT UNSW models, so the search
cannot alter either TTL field. Reports, against the standard (TTL-mutable) confirmatory arms:
  - coverage (M1), eligible class-2/3 violation rate (M2), joint success (M3),
  - TTL-referencing constraint violation counts (must be driven only by other-feature edits now),
  - immutable-mismatch on sttl/dttl (0 by construction - a sanity check).
Seeds 10,11,12 (primary + two). Needs the Stage-3A reruns present for the standard-arm comparison.
Checkpoint: skips a (det,arm,seed) whose npz already exists.
Usage: python exp_R2_5_ttl_fixed.py <RF|XGB|HGB>
"""
import os, sys, time, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace'); sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
import expbase as EB
from expcommon import metrics_from_array

ds = 'unsw'; det = sys.argv[1]
OUT = os.path.join(EB.EXP, 'R2_5_ttlfixed', 'raw'); os.makedirs(OUT, exist_ok=True)
REPRO = os.path.join(EB.EXP, 'repro', 'raw')
sp, feats, meta = EB.build_space(ds, extra_immutable=('sttl', 'dttl'))
model = EB.load_model(ds, det)
X0, c0, idx = EB.conf_cases(ds, feats)

ttl_cols = [i for i, cid in enumerate([c.id for c in sp.constraints]) if ('sttl' in cid or 'dttl' in cid)]


def ttl_viol_count(z):
    v = z['viol']
    return int(v[:, ttl_cols].any(1).sum()) if ttl_cols else 0


RUNS = [('A_typed', 10), ('B', 10), ('A_typed', 11), ('B', 11), ('A_typed', 12), ('B', 12)]
summary = {'ds': ds, 'det': det, 'n_ttl_constraints': len(ttl_cols), 'runs': {}}
for arm, seed in RUNS:
    fn = os.path.join(OUT, f'{ds}_{det}_{arm}_seed{seed}.npz')
    if os.path.exists(fn):
        print(f'skip {os.path.basename(fn)} (exists)', flush=True)
    else:
        t = time.time()
        r = EB.run_arm(sp, model, X0, arm, seed)
        info = EB.save_arm(fn, sp, feats, arm, seed, X0, c0, idx, r)
        print(f'{ds} {det} TTLfixed {arm} s{seed}: flip {info["flip"]:.3f} immMis {info["n_mis"]} [{time.time()-t:.0f}s]', flush=True)
    z = np.load(fn, allow_pickle=True)
    m = metrics_from_array(z, ds, is_C=False)
    rec = {'M1': m['M1'], 'M2': m['M2'], 'M3': m['M3'], 'ttl_violations': ttl_viol_count(z),
           'sttl_dttl_mismatch': int(z['immutable_mismatch_count'])}
    # standard-arm comparison (TTL mutable) from the Stage-3A rerun
    std = os.path.join(REPRO, f'{ds}_{det}_{arm}_seed{seed}.npz')
    if os.path.exists(std):
        zs = np.load(std, allow_pickle=True); ms = metrics_from_array(zs, ds, is_C=False)
        rec['standard'] = {'M1': ms['M1'], 'M2': ms['M2'], 'M3': ms['M3'], 'ttl_violations': ttl_viol_count(zs)}
    summary['runs'][f'{arm}_seed{seed}'] = rec

os.makedirs(os.path.join(EB.EXP, 'R2_5_ttlfixed'), exist_ok=True)
json.dump(summary, open(os.path.join(EB.EXP, 'R2_5_ttlfixed', f'{ds}_{det}_summary.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print(f'R2.5 {ds}_{det} done.', flush=True)
