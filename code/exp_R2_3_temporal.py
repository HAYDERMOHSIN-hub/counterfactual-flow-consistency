"""exp_R2_3_temporal.py - Reviewer R2.3: temporal-split evaluation with SEPARATE training (UNSW only).

Rationale and honesty note: the stratified random split used in the main study can let near-in-time
flows sit in both train and test. This experiment instead orders UNSW flows by their absolute start
time (Stime, retained from the raw parts before it is dropped for modelling), holds out the LATEST 30%
as a temporal test period, and trains three fresh detectors on the EARLIER 70% ONLY - no row from the
held-out period enters training, donor pool, ranges or ECDF reference. Counterfactuals are then
generated for held-out-period attack cases the temporal detectors still flag, and audited with the same
frozen constraint list. CIC-IDS2017 is deliberately NOT given a temporal split: after preprocessing it
carries no usable absolute timestamp in the modelled feature set, so a principled time order cannot be
reconstructed without fabricating one - stated as a transparent limitation rather than approximated.

Stages (each checkpointed; delete the file to force a redo):
  A build+encode temporal train/test   -> data/unsw_temporal_data.npz
  B train 3 detectors on earlier-only   -> data/unsw_temporal_{det}.pkl + perf
  C select held-out attack cases        -> data/unsw_temporal_conf_idx.npy
  D arms A_typed/B/C (seed 10) per det   -> experiments/R2_3_temporal/raw/
  E summary                              -> experiments/R2_3_temporal/summary.json
Usage: python exp_R2_3_temporal.py   (runs all stages; resumable)
"""
import os, sys, time, json, pickle, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace'); sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
import cfg
from common import SEED
from generator import Space, genetic_search, nearest_real
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from expcommon import metrics_from_array
import expbase as EB

ds = 'unsw'
TMP = os.path.join(EB.EXP, 'R2_3_temporal'); RAW = os.path.join(TMP, 'raw')
os.makedirs(RAW, exist_ok=True)
DATA_NPZ = os.path.join(cfg.DATA, 'unsw_temporal_data.npz')
META_J = os.path.join(cfg.DATA, 'unsw_temporal_meta.json')
HOLDOUT_FRAC = 0.30


def encode(df, feats, cat, codes):
    X = np.empty((len(df), len(feats)), dtype=np.float64)
    for j, f in enumerate(feats):
        if f in cat:
            X[:, j] = df[f].map(codes[f]).fillna(-1).to_numpy()
        else:
            X[:, j] = df[f].to_numpy(dtype=np.float64)
    return X


# ---- Stage A: build temporal split ----------------------------------------------------------
if not os.path.exists(DATA_NPZ):
    print('[A] building temporal split from data/unsw_raw.pkl ...', flush=True)
    df = pd.read_pickle(os.path.join(cfg.DATA, 'unsw_raw.pkl'))
    df.columns = [c.replace(' ', '') for c in df.columns]
    stime = pd.to_numeric(df['Stime'], errors='coerce').to_numpy()
    df['ct_ftp_cmd'] = pd.to_numeric(df['ct_ftp_cmd'].astype(str).str.strip().replace('', np.nan), errors='coerce')
    for c in ['ct_ftp_cmd', 'is_ftp_login', 'ct_flw_http_mthd']:
        df[c] = df[c].fillna(0)
    df['attack_cat'] = df['attack_cat'].astype('object').where(df['attack_cat'].notna(), 'Benign').astype(str).str.strip().replace({'Backdoors': 'Backdoor'})
    df['_stime'] = stime
    df = df.drop(columns=['srcip', 'sport', 'dstip', 'dsport', 'Stime', 'Ltime', 'source_file'])
    df = df.rename(columns={'Label': 'y'}); df['y'] = df['y'].astype('int8')
    cat = ['proto', 'state', 'service']
    for c in cat:
        df[c] = df[c].astype(str).str.strip()
    feats = [c for c in df.columns if c not in ('y', 'attack_cat', '_stime')]
    # dedup on features + labels (keep first), identical key-set to the main study
    df = df.drop_duplicates(subset=feats + ['y', 'attack_cat'])
    # order by absolute start time; rows with missing Stime sort last (kept, flagged)
    df = df.sort_values('_stime', kind='mergesort', na_position='last').reset_index(drop=True)
    n = len(df); cut = int(round(n * (1 - HOLDOUT_FRAC)))
    tr = df.iloc[:cut].reset_index(drop=True); te = df.iloc[cut:].reset_index(drop=True)
    codes = {c: {v: i for i, v in enumerate(sorted(tr[c].unique()))} for c in cat}
    Xtr = encode(tr, feats, cat, codes); Xte = encode(te, feats, cat, codes)
    np.savez_compressed(DATA_NPZ, Xtr=Xtr, ytr=tr['y'].to_numpy(), ctr=tr['attack_cat'].to_numpy().astype(str),
                        Xte=Xte, yte=te['y'].to_numpy(), cte=te['attack_cat'].to_numpy().astype(str),
                        stime_tr=tr['_stime'].to_numpy(dtype='float64'), stime_te=te['_stime'].to_numpy(dtype='float64'))
    meta = dict(features=feats, categorical=cat, codes=codes, n_train=len(tr), n_test=len(te),
                holdout_frac=HOLDOUT_FRAC, cut_stime=float(df['_stime'].iloc[cut]) if not np.isnan(df['_stime'].iloc[cut]) else None,
                train_classes=tr['attack_cat'].value_counts().to_dict(), test_classes=te['attack_cat'].value_counts().to_dict())
    json.dump(meta, open(META_J, 'w'), indent=1)
    print(f'[A] n={n} train(earlier)={len(tr)} test(held-out)={len(te)}', flush=True)

D = np.load(DATA_NPZ, allow_pickle=True)
meta = json.load(open(META_J)); feats = meta['features']
Xtr, ytr, ctr = D['Xtr'], D['ytr'], D['ctr']
Xte, yte, cte = D['Xte'], D['yte'], D['cte']

# ---- Stage B: train detectors on earlier-only -----------------------------------------------
rng = np.random.default_rng(SEED)
ben = np.flatnonzero(ytr == 0); att = np.flatnonzero(ytr == 1)
nben = min(300000, len(ben))
fit_idx = np.sort(np.concatenate([att, rng.choice(ben, nben, replace=False)]))
Xf, yf = Xtr[fit_idx], ytr[fit_idx]
model_defs = {
    'RF': lambda: RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_leaf=2, n_jobs=2, random_state=SEED),
    'XGB': lambda: xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, tree_method='hist', n_jobs=2, random_state=SEED),
    'HGB': lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=SEED)}
perf = {}
preds = {}
for name, mk in model_defs.items():
    mp = os.path.join(cfg.DATA, f'unsw_temporal_{name}.pkl')
    if os.path.exists(mp):
        m = pickle.load(open(mp, 'rb'))
    else:
        print(f'[B] training temporal {name} on {len(yf)} earlier rows ...', flush=True)
        m = mk(); m.fit(Xf, yf); pickle.dump(m, open(mp, 'wb'))
    try: m.set_params(n_jobs=1)
    except Exception: pass
    yh = m.predict(Xte); p = m.predict_proba(Xte)[:, 1]; preds[name] = yh
    tn, fp, fn, tp = confusion_matrix(yte, yh).ravel()
    perf[name] = dict(accuracy=accuracy_score(yte, yh), precision=precision_score(yte, yh, zero_division=0),
                      recall=recall_score(yte, yh, zero_division=0), f1=f1_score(yte, yh, zero_division=0),
                      fpr=fp / (fp + tn), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
    print(f'[B] temporal {name}: acc {perf[name]["accuracy"]:.4f} rec {perf[name]["recall"]:.4f} fpr {perf[name]["fpr"]:.4f}', flush=True)

# ---- Stage C: select held-out attack cases flagged by all three temporal detectors ----------
CONF = os.path.join(cfg.DATA, 'unsw_temporal_conf_idx.npy')
if os.path.exists(CONF):
    chosen = np.load(CONF)
else:
    elig = np.flatnonzero((yte == 1) & (preds['RF'] == 1) & (preds['XGB'] == 1) & (preds['HGB'] == 1))
    classes = sorted(set(cte[elig].tolist())); N = min(1000, len(elig))
    quota = int(np.ceil(N / max(1, len(classes)))); ch = []
    for c in classes:
        ii = elig[cte[elig] == c]; ch += list(rng.choice(ii, min(quota, len(ii)), replace=False))
    ch = np.array(ch)
    if len(ch) > N:
        ch = rng.choice(ch, N, replace=False)
    chosen = np.sort(ch); np.save(CONF, chosen)
print(f'[C] held-out confirmatory cases: {len(chosen)}', flush=True)

# ---- Stage D: arms on temporal detectors ----------------------------------------------------
sp = Space(ds, Xtr, ytr, feats, {'categorical': meta['categorical'], 'codes': meta['codes']}, SEED)
sp.immutable = np.array([f in EB.UNSW_IMMUTABLE for f in feats])
X0 = Xte[chosen]; c0 = cte[chosen]
RUNS = [('A_typed', 10), ('B', 10), ('C', 10)]
summary = {'ds': ds, 'holdout_frac': HOLDOUT_FRAC, 'n_cases': int(len(chosen)),
           'temporal_perf': perf, 'cut_stime': meta.get('cut_stime'), 'cells': {}}
for det in cfg.DETECTORS:
    model = pickle.load(open(os.path.join(cfg.DATA, f'unsw_temporal_{det}.pkl'), 'rb'))
    try: model.set_params(n_jobs=1)
    except Exception: pass
    summary['cells'][det] = {}
    for arm, seed in RUNS:
        fn = os.path.join(RAW, f'unsw_{det}_temporal_{arm}_seed{seed}.npz')
        if not os.path.exists(fn):
            t = time.time()
            r = nearest_real(sp, model, X0) if arm == 'C' else genetic_search(sp, model, X0, arm, seed)
            EB.save_arm(fn, sp, feats, arm, seed, X0, c0, chosen, r)
            print(f'[D] temporal unsw {det} {arm}: flip {(r["returned"] & (r["p_benign"]>0.5)).mean():.3f} [{time.time()-t:.0f}s]', flush=True)
        z = np.load(fn, allow_pickle=True)
        m = metrics_from_array(z, ds, is_C=(arm == 'C'))
        summary['cells'][det][arm] = {'M1': m['M1'], 'M2': m['M2'], 'M3': m['M3'],
                                      'M4_pct_med': float(np.median(z['d_pct'][z['flipped']])) if z['flipped'].any() else None}

json.dump(summary, open(os.path.join(TMP, 'summary.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('[E] R2.3 temporal done.', flush=True)
