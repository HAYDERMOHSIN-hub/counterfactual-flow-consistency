"""exp_R2_2_dice.py - Reviewer R2.2: independent counterfactual generator (DiCE) as a robustness check,
now across ALL detectors (RF/XGB/HGB) and BOTH datasets, on the confirmatory cases, audited with the
same frozen constraint list. This extends the original single-detector (XGB) external check.

DiCE methods: genetic and random (library defaults, one counterfactual per instance). Integer-typed
columns are passed as integers so DiCE keeps them integral. Immutable features (incl. the 26-Oct UNSW
mask) are held via features_to_vary. The audit, distances and five-state metrics use the identical
Space as every other arm, so DiCE results are directly comparable to arms A_typed/B/C.

N defaults to 200 confirmatory cases per cell (a deterministic seeded subsample of the 1,000), matching
the original external-check sample size; pass a larger N to widen it. Checkpoint: skips an existing npz.
Usage: python exp_R2_2_dice.py <cic|unsw> <RF|XGB|HGB> [N=200] [methods=genetic,random]
"""
import os, sys, time, warnings, json, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace'); sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
import expbase as EB
import cfg
from common import load
from generator import p_benign
import constraints as K
from expcommon import metrics_from_array
import dice_ml
try:
    import importlib.metadata as _im
    DICE_VER = _im.version('dice-ml')
except Exception:
    DICE_VER = getattr(dice_ml, '__version__', '?')

ds, det = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 200
methods = sys.argv[4].split(',') if len(sys.argv) > 4 else ['genetic', 'random']
OUT = os.path.join(EB.EXP, 'R2_2_dice', 'raw'); os.makedirs(OUT, exist_ok=True)

sp, feats, meta = EB.build_space(ds)
model = EB.load_model(ds, det)
Xtr, ytr, ctr, _, _ = load(ds, 'train')
rng = np.random.default_rng(EB.SEED)
fit_idx = np.load(os.path.join(cfg.DATA, f'{ds}_fit_idx.npy'))
bg = Xtr[rng.choice(fit_idx, 20000, replace=False)]
del Xtr
X0all, c0all, idxall = EB.conf_cases(ds, feats)
sub = np.sort(rng.permutation(len(idxall))[:min(N, len(idxall))])
X0, c0, idx = X0all[sub], c0all[sub], idxall[sub]

cat = [f for f in feats if f in meta['categorical']]
cont = [f for f in feats if f not in cat]
ints = set(K.CIC_INT if ds == 'cic' else K.UNSW_INT)


def frame(X):
    d = pd.DataFrame(X, columns=feats)
    for f in feats:
        if f in cat:
            d[f] = d[f].astype(int).astype(str)
        elif f in ints:
            d[f] = d[f].round().astype('int64')
    return d


class Wrap:
    def __init__(s, m): s.m = m; s.classes_ = m.classes_
    def _a(s, d): return d[feats].astype(float).to_numpy()
    def predict_proba(s, d): return s.m.predict_proba(s._a(d))
    def predict(s, d): return s.m.predict(s._a(d))


dbg = frame(bg); dbg['y'] = model.predict(bg)
prec = {f: (0 if f in ints else 6) for f in cont}
try:
    data = dice_ml.Data(dataframe=dbg, continuous_features=cont, outcome_name='y', continuous_features_precision=prec)
except TypeError:
    data = dice_ml.Data(dataframe=dbg, continuous_features=cont, outcome_name='y')
mm = dice_ml.Model(model=Wrap(model), backend='sklearn')
vary = [f for j, f in enumerate(feats) if not sp.immutable[j]]
pr = {f: [float(sp.lo[sp.ix[f]]), float(sp.hi[sp.ix[f]])] for f in cont if f in vary}

summary = {'ds': ds, 'det': det, 'N': int(len(sub)), 'dice_version': DICE_VER, 'methods': {}}
for meth in methods:
    fn = os.path.join(OUT, f'{ds}_{det}_DiCE-{meth}.npz')
    if os.path.exists(fn):
        print(f'skip {os.path.basename(fn)} (exists)', flush=True)
    else:
        exp = dice_ml.Dice(data, mm, method=meth); t = time.time()
        cf = X0.copy(); ret = np.zeros(len(X0), bool); errs = []
        for i in range(len(X0)):
            try:
                kw = dict(total_CFs=1, desired_class=0, features_to_vary=vary, permitted_range=pr, verbose=False)
                if meth == 'random':
                    kw['random_seed'] = 0
                r = exp.generate_counterfactuals(frame(X0[i:i + 1]), **kw)
                out = r.cf_examples_list[0].final_cfs_df
                if out is not None and len(out) > 0:
                    cf[i] = out[feats].astype(float).to_numpy()[0]; ret[i] = True
            except Exception as e:
                errs.append(type(e).__name__ + ': ' + str(e)[:80])
        pb = p_benign(model, cf); flipped = ret & (pb > 0.5)
        viol, app = sp.audit(cf)
        np.savez_compressed(fn, cf=cf, x0=X0, cls=c0, test_idx=idx, returned=ret, flipped=flipped, p_benign=pb,
                            viol=viol, app=app, oor=sp.out_of_range(cf), d_pct=sp.dist_pct(sp.pct(X0), sp.pct(cf)),
                            d_gower=sp.dist_gower(X0, cf), l0=sp.l0(X0, cf),
                            constraint_ids=np.array([c.id for c in sp.constraints]),
                            constraint_cls=np.array([c.cls for c in sp.constraints]),
                            seconds=time.time() - t, sub=sub, n_errors=len(errs),
                            errors=np.array(sorted(set(errs))), method=meth, train_row=np.array([]))
        v = viol.any(1)
        print(f'{ds} {det} DiCE-{meth}: returned {ret.mean():.3f} flip {flipped.mean():.3f} '
              f'viol|flip {v[flipped].mean() if flipped.any() else float("nan"):.3f} errors {len(errs)} [{time.time()-t:.0f}s]', flush=True)
    z = np.load(fn, allow_pickle=True)
    m = metrics_from_array(z, ds, is_C=True)   # DiCE may fail to return (like arm C: candidate may not exist)
    summary['methods'][meth] = {'returned': float(z['returned'].mean()), 'M1': m['M1'], 'M2': m['M2'], 'M3': m['M3'],
                                'n_errors': int(z['n_errors']),
                                'M4_pct_med': float(np.median(z['d_pct'][z['flipped']])) if z['flipped'].any() else None}

os.makedirs(os.path.join(EB.EXP, 'R2_2_dice'), exist_ok=True)
json.dump(summary, open(os.path.join(EB.EXP, 'R2_2_dice', f'{ds}_{det}_summary.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print(f'R2.2 {ds}_{det} done (dice {summary["dice_version"]}).', flush=True)
