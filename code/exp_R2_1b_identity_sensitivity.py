"""exp_R2_1b_identity_sensitivity.py - Reviewer R2.1 (second part): concentration of the eligible
class-2/3 violation rate (M2) on the definitional / derived-feature EQUALITY identities, with
denominators and Wilson 95% intervals.

For arm A_typed (seeds 10/11/12) per cell it recomputes M2 with and without the duplicate/derived
equality identities:
  CIC : CIC-A-avgseg=mean[Fwd], CIC-A-avgseg=mean[Bwd]  (Avg Segment Size == Packet Length Mean)
  UNSW: UNSW-A-smeansz, UNSW-A-dmeansz                   (mean == bytes / packets)
These are GENUINE consistency violations (a duplicate/derived column left unupdated); excluding them
is a SENSITIVITY, not a correction - the main result keeps the full denominator. Coverage M1 and joint
M3 are reported too (M1 unchanged; M3 rises by the excluded share). No model calls.

Inputs: the A_typed arrays for the six cells and seeds 10/11/12 (6 x 3 = 18 arrays). By default it reads
them from evidence/arrays/confirmatory (relative to the repository root); pass a different directory as
the first argument. Calculations, metric definitions and the excluded-identity list are unchanged.

Usage:
  python code/exp_R2_1b_identity_sensitivity.py [arrays_dir] [out.json] [--partial]
Defaults: arrays_dir = evidence/arrays/confirmatory ; out.json = R2_1b_identity_sensitivity.json
Full mode (default) REQUIRES all 18 arrays; any missing input prints a clear error and exits non-zero.
--partial allows a partial run (missing arrays are skipped and the output is marked partial); it is for
diagnostics only and must not be used as an acceptance check.
"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace'); sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
import expcommon as E

DATASETS = ['cic', 'unsw']
DETECTORS = ['RF', 'XGB', 'HGB']
SEEDS = [10, 11, 12]
DROP = {'cic': ['CIC-A-avgseg=mean[Fwd]', 'CIC-A-avgseg=mean[Bwd]'],
        'unsw': ['UNSW-A-smeansz', 'UNSW-A-dmeansz']}

args = [a for a in sys.argv[1:] if not a.startswith('--')]
PARTIAL = '--partial' in sys.argv[1:]
RAW = args[0] if len(args) > 0 else 'evidence/arrays/confirmatory'
OUT = args[1] if len(args) > 1 else 'R2_1b_identity_sensitivity.json'


def path_for(ds, det, seed):
    return os.path.join(RAW, f'{ds}_{det}_A_typed_seed{seed}.npz')


# Pre-flight: require all 18 inputs unless --partial.
expected = [(ds, det, s) for ds in DATASETS for det in DETECTORS for s in SEEDS]
missing = [path_for(ds, det, s) for ds, det, s in expected if not os.path.exists(path_for(ds, det, s))]
if missing and not PARTIAL:
    sys.stderr.write(
        'ERROR: %d of %d required A_typed arrays are missing under %r.\n'
        'This analysis needs the six cells x seeds 10/11/12 (18 arrays). Missing:\n  %s\n'
        'Pass the correct arrays directory as the first argument, or use --partial for a diagnostic '
        'partial run (not an acceptance check).\n' % (len(missing), len(expected), RAW, '\n  '.join(missing)))
    sys.exit(1)


def m2_excl(z, ds, drop):
    el, cls2 = E.elig_sets(ds)
    ids = [str(x) for x in z['constraint_ids']]; col = {c: i for i, c in enumerate(ids)}
    present = [d for d in drop if d in col]
    elig = [c for c in el if c in col and c not in drop]; eligc = [col[c] for c in elig]
    cls2c = [col[c] for c in cls2 if c in col and c not in drop]
    return E.states_for(z, elig, cls2c, eligc, is_C=False), present


rep = {'arm': 'A_typed', 'seeds': SEEDS, 'excluded_identities': DROP, 'partial': bool(missing and PARTIAL),
       'arrays_dir': RAW, 'cells': {}}
print('R2.1b - M2 concentration on definitional/derived identities (A_typed)')
print('cell      seed | FULL M2 s4/den [95%CI]      | EXCL M2 s4/den [95%CI]')
n_done = 0
for ds in DATASETS:
    for det in DETECTORS:
        cell = f'{ds}_{det}'; rep['cells'][cell] = {}
        for seed in SEEDS:
            fn = path_for(ds, det, seed)
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
            n_done += 1
            if seed == 10:
                print('%-9s %4d | %.3f %3d/%d [%.3f,%.3f] | %.3f %3d/%d [%.3f,%.3f]' % (
                    cell, seed, f['M2'], f['s4'], den, lo, hi, ex['M2'], ex['s4'], den2, lo2, hi2))

rep['n_combinations_computed'] = n_done
json.dump(rep, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('\nComputed %d of 18 combinations. Wrote %s%s' % (n_done, OUT, ' (PARTIAL)' if rep['partial'] else ''))
if n_done == 0:
    sys.stderr.write('ERROR: no combinations computed (no arrays found).\n'); sys.exit(1)
