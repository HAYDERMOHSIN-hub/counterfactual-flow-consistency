#!/usr/bin/env python3
"""verify_tables_3_5.py - RE-ANALYSE the full confirmatory outcome tables (Tables 3, 4 and 5)
directly from the saved per-instance arrays, at the primary seed 10. Nothing is copied from the
manuscript tables or any summary JSON; every figure below is computed from the arrays.

What it reproduces
  Table 3  Confirmatory outcomes by cell and arm: the five states s1-s5, the denominators, and
           M1 (coverage), M2 (eligible class-2/3 violation rate over converted), M3 (joint),
           M4 (median percentile/ECDF distance d_pct over converted candidates).
  Table 4  A_typed M2 with the 95% Wilson interval and the pre-specified 5% decision
           (supported if the lower bound > 5%, not supported if the upper bound < 5%, else
           inconclusive; a cell with < 100 converted is inconclusive by rule).
  Table 5  Paired families on seed 10, Holm-corrected within each family of six cells (m=6), with
           any cell having < 100 usable pairs/converted treated as a non-rejected hypothesis (p=1):
             F1  M1  A_typed vs B   exact McNemar (binomial on discordant pairs)
             F2  M4  A_typed vs B   Wilcoxon signed-rank over instances converted by both
             F3  M4  B      vs C    Wilcoxon signed-rank over instances converted by both

Definitions are the deposited ones (expcommon.states_for): for a generated arm an instance
"converts" when p_benign > 0.5 (the return flag is NOT used as the M2 denominator for generated
arms); for arm C conversion additionally requires that a nearest real benign flow was found
(returned). M2's denominator is the converted set s3+s4+s5, conditional on conversion.

Arm sourcing (explicit): A_typed and C are read from the confirmatory arrays; arm B is read from
the CORRECTED eligible-list arrays (manuscript Section 3.5, deviation 1), which is the arm-B source
the manuscript reports. This is re-analysis from saved arrays, not regeneration.

Inputs (defaults, relative to the repository root):
  confirmatory dir = evidence/arrays/confirmatory   (A_typed, A raw, C)
  corrected-B dir  = evidence/arrays/B_corrected     (B)
Usage:
  python code/verify_tables_3_5.py [confirmatory_dir] [corrected_B_dir] [out.json]
Any missing required array prints a clear error and exits non-zero; no partial silent success.
"""
import os
import sys
import json
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import expcommon as E

DATASETS = ['cic', 'unsw']
DETECTORS = ['RF', 'XGB', 'HGB']
SEED = 10
INCONCLUSIVE_N = 100           # <100 converted / usable pairs -> non-rejected (p=1), 26-Oct decision (f)
CRIT = 0.05                    # pre-specified M2 decision threshold
RAW = sys.argv[1] if len(sys.argv) > 1 else 'evidence/arrays/confirmatory'
RAW_B = sys.argv[2] if len(sys.argv) > 2 else 'evidence/arrays/B_corrected'
OUT = sys.argv[3] if len(sys.argv) > 3 else 'tables_3_5_verified.json'
ARMS = ['A_typed', 'B', 'C']
CELLS = [f'{ds}_{det}' for ds in DATASETS for det in DETECTORS]


def arm_path(ds, det, arm):
    base = RAW_B if arm == 'B' else RAW
    return os.path.join(base, f'{ds}_{det}_{arm}_seed{SEED}.npz')


def holm(pvals):
    """Holm step-down adjusted p-values (monotone), same routine as the deposited analysis."""
    pvals = np.asarray(pvals, float)
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    run = 0.0
    for rank, idx in enumerate(order):
        a = (m - rank) * pvals[idx]
        run = max(run, a)
        adj[idx] = min(run, 1.0)
    return adj


def mcnemar_p(a_succ, b_succ):
    """Exact McNemar: binomial test on the discordant pairs (b = A-only, c = B-only)."""
    b = int((a_succ & ~b_succ).sum())
    c = int((~a_succ & b_succ).sum())
    if b + c == 0:
        return 1.0, b, c
    return float(stats.binomtest(min(b, c), b + c, 0.5).pvalue), b, c


def wilcoxon_p(dA, dB):
    m = ~np.isnan(dA) & ~np.isnan(dB)
    if m.sum() < 1 or np.all(dA[m] == dB[m]):
        return 1.0, int(m.sum())
    try:
        return float(stats.wilcoxon(dA[m], dB[m]).pvalue), int(m.sum())
    except Exception:
        return 1.0, int(m.sum())


# ---- load every cell/arm once; keep states, conv mask, d_pct ----------------------------------
required = [arm_path(ds, det, a) for ds in DATASETS for det in DETECTORS for a in ARMS]
missing = [p for p in required if not os.path.exists(p)]
if missing:
    sys.stderr.write('ERROR: %d required arrays missing:\n  %s\n' % (len(missing), '\n  '.join(missing)))
    sys.exit(1)

cell_data = {}
for ds in DATASETS:
    for det in DETECTORS:
        cell = f'{ds}_{det}'
        cell_data[cell] = {}
        for arm in ARMS:
            z = np.load(arm_path(ds, det, arm), allow_pickle=True)
            m = E.metrics_from_array(z, ds, is_C=(arm == 'C'))
            m['d_pct'] = np.asarray(z['d_pct'], float)
            m['M4'] = float(np.median(m['d_pct'][m['conv']])) if m['conv'].any() else None
            cell_data[cell][arm] = m

rep = {
    'seed': SEED,
    'armB_source': 'corrected eligible-list (evidence/arrays/B_corrected)',
    'armAC_source': 'confirmatory (evidence/arrays/confirmatory)',
    'definitions': {
        'convert_generated': 'p_benign > 0.5 (return flag NOT used as M2 denominator)',
        'convert_armC': 'p_benign > 0.5 AND a nearest real benign flow was returned',
        'M1': '(s3+s4+s5)/N', 'M2': 's4/(s3+s4+s5)', 'M3': 's5/N',
        'M4': 'median d_pct over converted candidates',
        'inconclusive_rule': '<100 converted / usable pairs -> non-rejected (p=1)'},
    'table3_cells': {}, 'table4_M2_vs_5pct': {}, 'table5_families': {}}

# ---- Table 3 -----------------------------------------------------------------------------------
print('=== Table 3: confirmatory outcomes by cell and arm (seed 10) ===')
print('cell       arm       N   s1  s2  s3  s4  s5     M1     M2     M3     M4')
for cell in CELLS:
    rep['table3_cells'][cell] = {}
    for arm in ARMS:
        m = cell_data[cell][arm]
        m2 = None if m['M2'] is None else round(m['M2'], 3)
        m4 = None if m['M4'] is None else round(m['M4'], 5)
        rep['table3_cells'][cell][arm] = {
            'N': m['N'], 's1': m['s1'], 's2': m['s2'], 's3': m['s3'], 's4': m['s4'], 's5': m['s5'],
            'n_converted': m['n_conv'], 'M1': round(m['M1'], 3), 'M2': m2,
            'M3': round(m['M3'], 3), 'M4': m4}
        print('%-10s %-9s %4d %3d %3d %3d %3d %3d  %5.3f  %s  %5.3f  %s' % (
            cell, arm, m['N'], m['s1'], m['s2'], m['s3'], m['s4'], m['s5'], m['M1'],
            ('  -  ' if m2 is None else '%5.3f' % m2), m['M3'],
            ('  -   ' if m4 is None else '%6.5f' % m4)))

# ---- Table 4: A_typed M2 vs 5% (Wilson) --------------------------------------------------------
print('\n=== Table 4: A_typed M2 with 95% Wilson interval and 5% verdict ===')
print('cell        n_conv  s4    M2     wilson95            verdict')
for cell in CELLS:
    A = cell_data[cell]['A_typed']
    n, s4 = A['n_conv'], A['s4']
    lo, hi = E.wilson(s4, n)
    if n < INCONCLUSIVE_N:
        verdict = 'inconclusive (<100 converted)'
    elif lo is not None and lo > CRIT:
        verdict = 'supported (>5%)'
    elif hi is not None and hi < CRIT:
        verdict = 'not supported (<5%)'
    else:
        verdict = 'inconclusive'
    rep['table4_M2_vs_5pct'][cell] = {
        'n_converted': n, 'state4': s4, 'M2': None if A['M2'] is None else round(A['M2'], 4),
        'wilson95': [None if lo is None else round(lo, 4), None if hi is None else round(hi, 4)],
        'verdict': verdict}
    print('%-10s %5d %5d  %s  [%s, %s]  %s' % (
        cell, n, s4, ('  -  ' if A['M2'] is None else '%5.3f' % A['M2']),
        '----' if lo is None else '%.4f' % lo, '----' if hi is None else '%.4f' % hi, verdict))

# ---- Table 5: paired families, Holm m=6 --------------------------------------------------------
fam_specs = [
    ('F1_M1_Atyped_vs_B', 'M1 A_typed vs B, exact McNemar, all evaluated instances'),
    ('F2_M4_Atyped_vs_B', 'M4 A_typed vs B, Wilcoxon, converted in both arms'),
    ('F3_M4_B_vs_C', 'M4 B vs C, Wilcoxon, converted in both arms')]
print('\n=== Table 5: paired families (seed 10, Holm m=6) ===')
for fam, desc in fam_specs:
    info = []
    raw = []
    for cell in CELLS:
        A = cell_data[cell]['A_typed']
        B = cell_data[cell]['B']
        C = cell_data[cell]['C']
        if fam.startswith('F1'):
            incon = (A['n_conv'] < INCONCLUSIVE_N) or (B['n_conv'] < INCONCLUSIVE_N)
            if incon:
                p, b, c, note = 1.0, None, None, 'inconclusive (<100 converted) -> p=1'
            else:
                p, b, c = mcnemar_p(A['conv'], B['conv'])
                note = ''
            info.append({'cell': cell, 'p': p, 'b_Aonly': b, 'c_Bonly': c,
                         'inconclusive': incon, 'note': note})
        elif fam.startswith('F2'):
            both = A['conv'] & B['conv']
            n = int(both.sum())
            incon = n < INCONCLUSIVE_N
            if incon:
                p, note = 1.0, 'inconclusive (<100 usable pairs) -> p=1'
            else:
                p, _ = wilcoxon_p(A['d_pct'][both], B['d_pct'][both])
                note = ''
            info.append({'cell': cell, 'p': p, 'n_pairs': n, 'inconclusive': incon,
                         'median_A': float(np.median(A['d_pct'][both])) if n else None,
                         'median_B': float(np.median(B['d_pct'][both])) if n else None, 'note': note})
        else:
            both = B['conv'] & C['conv']
            n = int(both.sum())
            incon = n < INCONCLUSIVE_N
            if incon:
                p, note = 1.0, 'inconclusive (<100 usable pairs) -> p=1'
            else:
                p, _ = wilcoxon_p(B['d_pct'][both], C['d_pct'][both])
                note = ''
            info.append({'cell': cell, 'p': p, 'n_pairs': n, 'inconclusive': incon,
                         'median_B': float(np.median(B['d_pct'][both])) if n else None,
                         'median_C': float(np.median(C['d_pct'][both])) if n else None, 'note': note})
        raw.append(info[-1]['p'])
    adj = holm(raw)
    for i, x in enumerate(info):
        x['p_holm'] = float(adj[i])
        x['sig_0.05'] = bool(adj[i] < 0.05)
    rep['table5_families'][fam] = {'description': desc, 'm': 6, 'cells': info,
                                   'n_inconclusive': sum(1 for x in info if x['inconclusive'])}
    print('  %s (inconclusive=%d):' % (fam, rep['table5_families'][fam]['n_inconclusive']))
    for x in info:
        print('     %-10s p=%.3g  p_holm=%.3g  sig=%s  %s' % (
            x['cell'], x['p'], x['p_holm'], x['sig_0.05'], x.get('note', '')))

json.dump(rep, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('\nWrote %s' % OUT)
