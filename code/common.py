"""common.py - shared loading utilities."""
import json, numpy as np, pandas as pd, sys, os
SEED = 20261007

def load(ds, split):
    meta = json.load(open(f'data/{ds}_meta.json'))
    df = pd.read_pickle(f'data/{ds}_{split}.pkl')
    feats, cat = meta['features'], meta['categorical']
    if 'codes' not in meta:
        tr = df if split == 'train' else pd.read_pickle(f'data/{ds}_train.pkl')
        meta['codes'] = {c: {v: i for i, v in enumerate(sorted(tr[c].unique()))} for c in cat}
        json.dump(meta, open(f'data/{ds}_meta.json', 'w'), indent=1)
    X = np.empty((len(df), len(feats)), dtype=np.float64)
    for j, f in enumerate(feats):
        X[:, j] = df[f].map(meta['codes'][f]).fillna(-1).to_numpy() if f in cat else df[f].to_numpy(dtype=np.float64)
    return X, df['y'].to_numpy(), df['attack_cat'].to_numpy().astype(str), feats, meta
