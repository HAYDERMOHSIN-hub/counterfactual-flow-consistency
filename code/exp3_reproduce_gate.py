"""exp3_reproduce_gate.py - STAGE 3A (gate): verify the arms re-run on the REBUILT models reproduce
the saved confirmatory arrays, so later experiments stand on a fair baseline.

Comparison (per cell x arm x seed), reference = original_48 for A_typed/A/C, corrected_B_18 for B:
  pre-defined PASS criteria (NOT tuned):
    |dM1| <= 0.005, |dM2| <= 0.005, |dM3| <= 0.005  (M2 None==None counts as match)
    per-case flip agreement >= 0.99
  diagnostics (reported, not gated): identical-CF fraction, max |dp_benign| on identical CFs.
Also emits the R1.4 query-budget table (expects 1260 = 60 + 40x30 for every genetic arm; 0 for C).
Writes logs/reproduce_gate.json. Exit 0 = all pass -> proceed to new experiments, 3 = material difference.
"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace'); sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
import cfg
from expcommon import metrics_from_array
from generator import POP, GENS, ELITE

RERUN = os.path.join(cfg.ROOT, 'experiments', 'repro', 'raw')
# Reference arrays live in the repository at evidence/arrays/{confirmatory,B_corrected}. Older
# deposited layouts used reference/{original_48,corrected_B_18}; both are accepted. Override with
# PAPER4_REF_CONF / PAPER4_REF_B if the arrays are elsewhere.
def _first_existing(cands):
    for c in cands:
        if os.path.isdir(c):
            return c
    return cands[0]

REF_CONF = os.environ.get('PAPER4_REF_CONF', _first_existing([
    os.path.join(cfg.ROOT, 'evidence', 'arrays', 'confirmatory'),
    os.path.join(cfg.REF, 'original_48')]))
REF_B = os.environ.get('PAPER4_REF_B', _first_existing([
    os.path.join(cfg.ROOT, 'evidence', 'arrays', 'B_corrected'),
    os.path.join(cfg.REF, 'corrected_B_18')]))
EXPECTED_Q = POP + GENS * (POP - ELITE)   # 60 + 40*30 = 1260
DM = 0.005; AG = 0.99

RUNS = [('A_typed', 10), ('B', 10), ('C', 10), ('A', 10),
        ('A_typed', 11), ('B', 11), ('A_typed', 12), ('B', 12)]
CELLS = [(d, m) for d in cfg.DATASETS for m in cfg.DETECTORS]


def ref_path(ds, det, arm, seed):
    base = REF_B if arm == 'B' else REF_CONF
    return os.path.join(base, f'{ds}_{det}_{arm}_seed{seed}.npz')


def dmetric(a, b):
    if a is None and b is None:
        return 0.0
    if a is None or b is None:
        return 1.0
    return abs(a - b)


rep = {'stage': 'reproduce_gate', 'criteria': {'dM': DM, 'flip_agreement': AG, 'expected_queries': EXPECTED_Q},
       'reference_dirs': {'confirmatory': REF_CONF, 'corrected_B': REF_B}, 'rerun_dir': RERUN,
       'cells': {}, 'query_budget': [], 'all_pass': True, 'missing': []}

for ds, det in CELLS:
    cell = f'{ds}_{det}'; rep['cells'][cell] = {}
    for arm, seed in RUNS:
        tag = f'{arm}_seed{seed}'
        pr = os.path.join(RERUN, f'{ds}_{det}_{arm}_seed{seed}.npz'); rf = ref_path(ds, det, arm, seed)
        if not os.path.exists(pr) or not os.path.exists(rf):
            rep['missing'].append(f'{cell}:{tag} (rerun={os.path.exists(pr)}, ref={os.path.exists(rf)})')
            rep['all_pass'] = False
            rep['cells'][cell][tag] = {'pass': False, 'note': 'array missing'}
            continue
        zr = np.load(pr, allow_pickle=True); zf = np.load(rf, allow_pickle=True)
        is_C = (arm == 'C')
        mr = metrics_from_array(zr, ds, is_C); mf = metrics_from_array(zf, ds, is_C)
        d1 = dmetric(mr['M1'], mf['M1']); d2 = dmetric(mr['M2'], mf['M2']); d3 = dmetric(mr['M3'], mf['M3'])
        flip_ag = float((zr['flipped'] == zf['flipped']).mean())
        same_cf = np.all(np.isclose(zr['cf'], zf['cf'], rtol=0, atol=1e-6), axis=1)
        cf_frac = float(same_cf.mean())
        maxdp = float(np.abs(zr['p_benign'][same_cf] - zf['p_benign'][same_cf]).max()) if same_cf.any() else 0.0
        ok = (d1 <= DM and d2 <= DM and d3 <= DM and flip_ag >= AG)
        rec = {'pass': bool(ok), 'dM1': round(d1, 5), 'dM2': round(d2, 5), 'dM3': round(d3, 5),
               'flip_agreement': round(flip_ag, 4), 'identical_cf_frac': round(cf_frac, 4),
               'max_dp_on_identical': maxdp,
               'rerun': {'M1': mr['M1'], 'M2': mr['M2'], 'M3': mr['M3']},
               'ref': {'M1': mf['M1'], 'M2': mf['M2'], 'M3': mf['M3']}}
        rep['cells'][cell][tag] = rec
        if not ok:
            rep['all_pass'] = False
        q = int(zr['queries']) if 'queries' in zr.files else 0
        rep['query_budget'].append({'cell': cell, 'arm': arm, 'seed': seed, 'queries_per_instance': q,
                                     'expected': 0 if is_C else EXPECTED_Q,
                                     'ok': (q == 0 if is_C else q == EXPECTED_Q)})

q_ok = all(x['ok'] for x in rep['query_budget']) if rep['query_budget'] else False
rep['R1_4_query_budget_ok'] = q_ok
rep['R1_4_formula'] = f'POP({POP}) + GENS({GENS}) x children({POP-ELITE}) = {EXPECTED_Q} proba evaluations per instance per genetic run'

os.makedirs(cfg.LOGS, exist_ok=True)
json.dump(rep, open(os.path.join(cfg.LOGS, 'reproduce_gate.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('=' * 70); print('STAGE 3A - REPRODUCE CONFIRMATORY ARMS ON REBUILT MODELS'); print('=' * 70)
for cell in rep['cells']:
    for tag, rec in rep['cells'][cell].items():
        t = 'PASS' if rec.get('pass') else 'FAIL'
        if 'dM1' in rec:
            print(f"  [{t}] {cell} {tag}: dM1={rec['dM1']} dM2={rec['dM2']} dM3={rec['dM3']} "
                  f"flipAgree={rec['flip_agreement']} sameCF={rec['identical_cf_frac']}")
        else:
            print(f"  [{t}] {cell} {tag}: {rec.get('note')}")
print(f"\nR1.4 query budget: {rep['R1_4_formula']}  -> all genetic arms == {EXPECTED_Q}: {q_ok}")
if rep['missing']:
    print('\nMissing arrays:', rep['missing'])
print('\nOVERALL:', 'ALL PASS -> proceed to new experiments' if rep['all_pass'] else
      'MATERIAL DIFFERENCE -> STOP; see logs/reproduce_gate.json')
sys.exit(0 if rep['all_pass'] else 3)
