"""exp_R2_1_dedup.py - Reviewer R2.1 (model-dependent part): de-duplication sensitivity of the
confirmatory metrics, computed over the Stage-3A reruns (on the REBUILT models) so no new generation
is needed. For each cell and arm (primary seed 10) it:
  - counts confirmatory cases that are exact feature-duplicates of another confirmatory case,
  - counts arm-C returned donors that coincide (same selected benign flow) across cases,
  - recomputes M1/M2/M3 after dropping duplicate-identity confirmatory cases (keep-first),
  - reports the metric deltas vs the full sample.
Pairs with the already-computed model-free secondary (near-duplicate exclusion from the M2 denominator).
Reads experiments/repro/raw. Writes experiments/R2_1_dedup/summary.json. Fast (no model calls).
Usage: python exp_R2_1_dedup.py
"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace'); sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
import cfg
from expcommon import cols_for, states_for

RERUN = os.path.join(cfg.ROOT, 'experiments', 'repro', 'raw')
OUT = os.path.join(cfg.ROOT, 'experiments', 'R2_1_dedup'); os.makedirs(OUT, exist_ok=True)
CELLS = [(d, m) for d in cfg.DATASETS for m in cfg.DETECTORS]
ARMS = ['A_typed', 'B', 'C', 'A']


def metrics_subset(z, ds, is_C, keep):
    """states_for on a boolean keep-mask over cases (re-slice the stored arrays)."""
    el, cls2, elig_cols, cls2_cols = cols_for(z, ds)
    zz = {'p_benign': z['p_benign'][keep], 'returned': z['returned'][keep], 'viol': z['viol'][keep],
          'train_row': (z['train_row'][keep] if z['train_row'].size == len(z['p_benign']) else np.zeros(0)),
          'files': ['p_benign', 'returned', 'viol', 'train_row']}
    return states_for(zz, el, cls2_cols, elig_cols, is_C)


summary = {'cells': {}}
for ds, det in CELLS:
    cell = f'{ds}_{det}'; summary['cells'][cell] = {}
    # duplicate-identity flags on the ORIGINAL cases (x0 identical across confirmatory rows)
    anyf = os.path.join(RERUN, f'{ds}_{det}_A_typed_seed10.npz')
    if not os.path.exists(anyf):
        summary['cells'][cell]['note'] = 'rerun missing'; continue
    x0 = np.load(anyf, allow_pickle=True)['x0']
    _, uniq_idx, inv, counts = np.unique(x0, axis=0, return_index=True, return_inverse=True, return_counts=True)
    is_dup = counts[inv] > 1                       # case shares its exact feature vector with another
    keep_first = np.zeros(len(x0), bool); keep_first[np.sort(uniq_idx)] = True
    summary['cells'][cell]['n_cases'] = int(len(x0))
    summary['cells'][cell]['n_exact_dup_cases'] = int(is_dup.sum())
    summary['cells'][cell]['n_unique_identities'] = int(len(uniq_idx))
    for arm in ARMS:
        fn = os.path.join(RERUN, f'{ds}_{det}_{arm}_seed10.npz')
        if not os.path.exists(fn):
            continue
        z = np.load(fn, allow_pickle=True); is_C = (arm == 'C')
        full = metrics_subset(z, ds, is_C, np.ones(len(x0), bool))
        dd = metrics_subset(z, ds, is_C, keep_first)
        rec = {'full': {'M1': full['M1'], 'M2': full['M2'], 'M3': full['M3'], 'n': full['N']},
               'dedup_identities': {'M1': dd['M1'], 'M2': dd['M2'], 'M3': dd['M3'], 'n': dd['N']}}
        if is_C:
            tr = z['train_row']
            if tr.size == len(x0):
                ret = z['returned']; used = tr[ret & (tr >= 0)]
                rec['arm_C_distinct_donors_used'] = int(len(set(used.tolist())))
                rec['arm_C_returned'] = int(ret.sum())
        summary['cells'][cell][arm] = rec

json.dump(summary, open(os.path.join(OUT, 'summary.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('=' * 60); print('R2.1 DE-DUPLICATION SENSITIVITY (model-dependent, over rebuilt reruns)'); print('=' * 60)
for cell, rec in summary['cells'].items():
    if 'n_cases' in rec:
        print(f"  {cell}: {rec['n_exact_dup_cases']}/{rec['n_cases']} exact-duplicate cases, "
              f"{rec['n_unique_identities']} unique identities")
print('\nWrote experiments/R2_1_dedup/summary.json', flush=True)
