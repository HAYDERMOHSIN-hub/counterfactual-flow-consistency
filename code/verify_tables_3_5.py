"""verify_tables_3_5.py - RE-ANALYSE the confirmatory outcomes (Tables 3-5) directly from the saved
per-instance arrays. Recomputes the five-state counts and M1/M2/M3 for each dataset x detector cell and
arm (A_typed, B, C) at the primary seed 10, using the deposited metric definitions
(expcommon.states_for); nothing is copied from the manuscript tables or any summary JSON.

Arm sourcing (stated explicitly): A_typed and C are read from the confirmatory arrays; arm B is read
from the CORRECTED eligible-list arrays (manuscript Section 3.5, deviation 1), which is the arm-B source
the manuscript reports. The denominators, five-state definitions and the eligible constraint basis are
the deposited ones; this is re-analysis from saved arrays, not regeneration.

Inputs (default, relative to the repository root):
  confirmatory dir = evidence/arrays/confirmatory   (A_typed, A raw, C)
  corrected-B dir  = evidence/arrays/B_corrected     (B)
Usage:
  python code/verify_tables_3_5.py [confirmatory_dir] [corrected_B_dir] [out.json]
Any missing required array prints a clear error and exits non-zero.
"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import expcommon as E

DATASETS = ['cic', 'unsw']
DETECTORS = ['RF', 'XGB', 'HGB']
SEED = 10
RAW = sys.argv[1] if len(sys.argv) > 1 else 'evidence/arrays/confirmatory'
RAW_B = sys.argv[2] if len(sys.argv) > 2 else 'evidence/arrays/B_corrected'
OUT = sys.argv[3] if len(sys.argv) > 3 else 'tables_3_5_verified.json'


def arm_path(ds, det, arm):
    base = RAW_B if arm == 'B' else RAW
    return os.path.join(base, f'{ds}_{det}_{arm}_seed{SEED}.npz')


ARMS = ['A_typed', 'B', 'C']
required = [arm_path(ds, det, a) for ds in DATASETS for det in DETECTORS for a in ARMS]
missing = [p for p in required if not os.path.exists(p)]
if missing:
    sys.stderr.write('ERROR: %d required arrays missing:\n  %s\n' % (len(missing), '\n  '.join(missing)))
    sys.exit(1)

rep = {'seed': SEED, 'armB_source': 'corrected eligible-list', 'cells': {}}
print('Tables 3-5 re-analysis from saved arrays (seed 10). M1=coverage, M2=eligible class-2/3 '
      'violation rate over converted, M3=joint.')
print('cell       arm       N   s1  s2  s3  s4  s5     M1     M2     M3')
for ds in DATASETS:
    for det in DETECTORS:
        cell = f'{ds}_{det}'; rep['cells'][cell] = {}
        for arm in ARMS:
            z = np.load(arm_path(ds, det, arm), allow_pickle=True)
            m = E.metrics_from_array(z, ds, is_C=(arm == 'C'))
            m2 = None if m['M2'] is None else round(m['M2'], 3)
            rep['cells'][cell][arm] = {'N': m['N'], 's1': m['s1'], 's2': m['s2'], 's3': m['s3'],
                                       's4': m['s4'], 's5': m['s5'], 'M1': round(m['M1'], 3),
                                       'M2': m2, 'M3': round(m['M3'], 3)}
            print('%-10s %-9s %4d %3d %3d %3d %3d %3d  %5.3f  %s  %5.3f' % (
                cell, arm, m['N'], m['s1'], m['s2'], m['s3'], m['s4'], m['s5'],
                m['M1'], ('  -  ' if m2 is None else '%5.3f' % m2), m['M3']))

json.dump(rep, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('\nWrote %s' % OUT)
