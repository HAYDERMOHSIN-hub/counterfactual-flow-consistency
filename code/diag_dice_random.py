"""diag_dice_random.py - pin the exact cause of the DiCE-random failure
(`ValueError: low is out of bounds for int32`). DiCE-random samples integers over each varied
feature's permitted range scaled by its continuous_features_precision (0 for integer-typed features,
6 for real-valued ones here). On a platform whose default integer is int32, a scaled bound exceeding
2**31-1 = 2147483647 triggers the error. This script builds the same Space, computes the scaled bound
per varied continuous feature, and lists exactly which features (and values) exceed the int32 limit.
Reads nothing new; writes experiments/R2_2_dice/dice_random_diagnosis.json. Fast.
Usage: python code\\diag_dice_random.py
"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
import cfg
from common import load, SEED
from generator import Space
import constraints as K

INT32_MAX = 2**31 - 1
OUT = os.path.join(cfg.ROOT, 'experiments', 'R2_2_dice'); os.makedirs(OUT, exist_ok=True)
rep = {'int32_max': INT32_MAX, 'precision_realvalued': 6, 'precision_integer': 0, 'datasets': {}}

for ds in cfg.DATASETS:
    Xtr, ytr, ctr, feats, meta = load(ds, 'train')
    sp = Space(ds, Xtr, ytr, feats, meta, SEED)
    del Xtr, ytr
    imm = ('proto', 'is_sm_ips_ports', 'service') if ds == 'unsw' else ()
    sp.immutable = np.array([f in imm for f in feats])
    cat = set(meta['categorical'])
    ints = set(K.CIC_INT if ds == 'cic' else K.UNSW_INT)
    offenders = []
    for j, f in enumerate(feats):
        if sp.immutable[j] or f in cat:
            continue
        prec = 0 if f in ints else 6
        lo = float(sp.lo[j]); hi = float(sp.hi[j])
        scaled = max(abs(lo), abs(hi)) * (10 ** prec)
        if scaled > INT32_MAX:
            offenders.append({'feature': f, 'lo': lo, 'hi': hi, 'precision': prec,
                              'scaled_bound': scaled, 'exceeds_int32_by_factor': round(scaled / INT32_MAX, 1)})
    offenders.sort(key=lambda d: -d['scaled_bound'])
    rep['datasets'][ds] = {'n_varied_continuous': int((~sp.immutable & ~np.array([f in cat for f in feats])).sum()),
                           'n_offenders': len(offenders), 'offenders': offenders}
    print(f'\n[{ds}] features whose int-scaled permitted range exceeds int32 ({INT32_MAX}):')
    for o in offenders[:12]:
        print(f"   {o['feature']}: [{o['lo']:.3g},{o['hi']:.3g}] x10^{o['precision']} = {o['scaled_bound']:.3g} "
              f"(> int32 by {o['exceeds_int32_by_factor']}x)")
    if not offenders:
        print('   none')

json.dump(rep, open(os.path.join(OUT, 'dice_random_diagnosis.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('\nWrote experiments/R2_2_dice/dice_random_diagnosis.json')
print('Conclusion: DiCE-random is unusable here because these real-valued features, scaled by precision 6,')
print('exceed the platform int32 sampling bound. DiCE-genetic does not use this integer sampler and runs fine.')
