"""exp_R2_1b_identity_sensitivity.py - Reviewer R2.1 (second part): concentration of the eligible
class-2/3 violation rate (M2) on the definitional / derived-feature EQUALITY identities, with
denominators and Wilson 95% intervals (reviewer R2.3/general: evidence-specific reporting).

For arm A_typed (seeds 10/11/12) per cell it recomputes M2 with and without the duplicate/derived
equality identities:
  CIC : CIC-A-avgseg=mean[Fwd], CIC-A-avgseg=mean[Bwd]  (Avg Segment Size == Packet Length Mean)
  UNSW: UNSW-A-smeansz, UNSW-A-dmeansz                   (mean == bytes / packets)
These are GENUINE consistency violations (a duplicate/derived column left unupdated); excluding them
is a SENSITIVITY, not a correction - the main result keeps the full denominator. Coverage M1 and joint
M3 are reported too (M1 unchanged; M3 rises by the excluded share). Reads experiments/repro/raw; writes
experiments/R2_1_dedup/identity_sensitivity.json. Fast (no model calls).
Usage: python exp_R2_1b_identity_sensitivity.py
"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace'); sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
import cfg
import expcommon as E

RERUN = os.path.join(cfg.ROOT, 'experiments', 'repro', 'raw')
OUT = os.path.join(cfg.ROOT, 'experiments', 'R2_1_dedup'); os.makedirs(OUT, exist_ok=True)
DROP = {'cic': ['CIC-A-avgseg=mean[Fwd]', 'CIC-A-avgseg=mean[Bwd]'],
        'unsw': ['UNSW-A-smeansz', 'UNSW-A-dmeansz']}
SEEDS = [10, 11, 12]


def m2_excl(z, ds, drop):
    el, cls2 = E.elig_sets(ds)
    ids = [str(x) for x in z['constraint_ids']]; col = {c: i for i, c in enumerate(ids)}
    present = [d for d in drop if d in col]
    elig = [c for c in el if c in col and c not in drop]; eligc = [col[c] for c in elig]
    cls2c = [col[c] for c in cls2 if c in col and c not in drop]
    return E.states_for(z, elig, cls2c, eligc, is_C=False), present


rep = {'arm': 'A_typed', 'seeds': SEEDS, 'excluded_identities': DROP, 'cells': {}}
print('R2.1b - M2 concentration on definitional/derived identities (A_typed)')
print('cell      seed | FULL M2 s4/den [95%CI]      | EXCL M2 s4/den [95%CI]')
for ds in cfg.DATASETS:
    for det in cfg.DETECTORS:
        cell = f'{ds}_{det}'; rep['cells'][cell] = {}
        for seed in SEEDS:
            fn = os.path.join(RERUN, f'{ds}_{det}_A_typed_seed{seed}.npz')
            if not os.path.exists(fn):
                rep['cells'][cell][str(seed)] = {'note': 'array missing'}; continue
            z = np.load(fn, allow_pickle=True)
            f = E.metrics_from_array(z, ds, is_C=False)
            den = f['s3'] + f['s4'] + f['s5']; lo, hi = E.wilson(f['s4'], den)
            ex, present = m2_excl(z, ds, DROP[ds]); den2 = ex['s3'] + ex['s4'] + ex['s5']; lo2, hi2 = E.wilson(ex['s4'], den2)
            rep['cells'][cell][str(seed)] = {
                'full': {'M2': f['M2'], 's4': f['s4'], 'denominator': den, 'wilson95': [lo, hi], 'M1': f['M1'], 'M3': f['M3']},
                'excl_identities': {'M2': ex['M2'], 's4': ex['s4'], 'denominator': den2, 'wilson95': [lo2, hi2], 'M1': ex['M1'], 'M3': ex['M3']},
                'excluded_present': present}
            if seed == 10:
                print('%-9s %4d | %.3f %3d/%d [%.3f,%.3f] | %.3f %3d/%d [%.3f,%.3f]' % (
                    cell, seed, f['M2'], f['s4'], den, lo, hi, ex['M2'], ex['s4'], den2, lo2, hi2))

json.dump(rep, open(os.path.join(OUT, 'identity_sensitivity.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('\nWrote experiments/R2_1_dedup/identity_sensitivity.json')
