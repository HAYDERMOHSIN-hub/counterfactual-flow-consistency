"""04_train_models.py - Fit the three detectors, report test performance, select the 1,000 original instances."""
import sys, json, pickle, numpy as np, pandas as pd
sys.path.insert(0, 'code')
from common import load, SEED
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
ds = sys.argv[1]
Xtr, ytr, ctr, feats, meta = load(ds, 'train')
rng = np.random.default_rng(SEED)
ben = np.flatnonzero(ytr == 0); att = np.flatnonzero(ytr == 1)
fit_idx = np.sort(np.concatenate([att, rng.choice(ben, 300000, replace=False)]))
np.save(f'data/{ds}_fit_idx.npy', fit_idx)
Xf, yf = Xtr[fit_idx], ytr[fit_idx]
models = {
    'RF': RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_leaf=2, n_jobs=2, random_state=SEED),
    'XGB': xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, tree_method='hist', n_jobs=2, random_state=SEED),
    'HGB': HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=SEED),
}
Xte, yte, cte, _, _ = load(ds, 'test')
rows = []; pred = {}
for name, m in models.items():
    m.fit(Xf, yf); p = m.predict_proba(Xte)[:, 1]; yh = (p >= 0.5).astype(int); pred[name] = yh
    tn, fp, fn, tp = confusion_matrix(yte, yh).ravel()
    rows.append(dict(dataset=ds, model=name, n_fit=len(yf), n_test=len(yte), accuracy=accuracy_score(yte, yh), precision=precision_score(yte, yh),
                     recall=recall_score(yte, yh), f1=f1_score(yte, yh), roc_auc=roc_auc_score(yte, p), fpr=fp / (fp + tn), tn=tn, fp=fp, fn=fn, tp=tp))
    pickle.dump(m, open(f'data/{ds}_{name}.pkl', 'wb')); print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(f'results/{ds}_detector_performance.csv', index=False)
# originals: test attacks detected by all three models; equal allocation across classes
elig = np.flatnonzero((yte == 1) & (pred['RF'] == 1) & (pred['XGB'] == 1) & (pred['HGB'] == 1))
N = 1000; classes = sorted(set(cte[elig])); quota = int(np.ceil(N / len(classes))); chosen = []
for c in classes:
    idx = elig[cte[elig] == c]; chosen += list(rng.choice(idx, min(quota, len(idx)), replace=False))
rest = np.setdiff1d(elig, chosen)
if len(chosen) < N: chosen += list(rng.choice(rest, N - len(chosen), replace=False))
chosen = np.sort(np.array(chosen[:N]) if len(chosen) > N else np.array(chosen))
if len(chosen) > N: chosen = np.sort(rng.choice(chosen, N, replace=False))
np.save(f'data/{ds}_orig_idx.npy', chosen)
print('eligible', len(elig), 'of', int((yte == 1).sum()), 'chosen', len(chosen)); print(pd.Series(cte[chosen]).value_counts())
pd.Series(cte[chosen]).value_counts().rename('n').to_csv(f'results/{ds}_originals_by_class.csv')
