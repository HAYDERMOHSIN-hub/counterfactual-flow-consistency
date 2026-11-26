"""R1.5 verification: four hand-built cases proving the implemented metric definition
(expcommon.states_for). For a GENERATED arm every instance has a stored candidate (exists=True),
so a converted, eligibly-violating, NOT-returned candidate enters s4 (M2 numerator). For arm C a
no-match case is s1 (in N for M1/M3, excluded from the conversion-conditional M2 denominator)."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '03_code', 'code'))
import expcommon as E
el, cls2 = E.elig_sets('cic'); cid = sorted(cls2)[0]
z = {'p_benign': np.array([0.9, 0.3, 0.8, 0.7]), 'returned': np.array([False, False, True, True]),
     'constraint_ids': np.array([cid]), 'viol': np.array([[True],[False],[False],[True]]),
     'train_row': np.zeros(0), 'files': ['p_benign','returned','constraint_ids','viol','train_row']}
g = E.states_for(z, list(el), [0], [0], is_C=False)
c = E.states_for(z, list(el), [0], [0], is_C=True)
print('GENERATED arm: N=4  M1=%.2f  M2=%.3f (s4=%d)  M3=%.2f' % (g['M1'], g['M2'], g['s4'], g['M3']))
print('  case1 (converted, class-2/3 violation, returned=False) counted in s4:', g['s4'] == 2)
print('ARM C: M1=%.2f (s1 no-match=%d within N; excluded from M2 denom)' % (c['M1'], c['s1']))
assert g['s4'] == 2 and c['s1'] == 2 and c['M1'] == 0.5
print('PASS: definition matches the documented behaviour.')
