"""02_clean_split.py - Minimal documented cleaning, duplicate removal, stratified 70/30 split.
Writes data/{cic,unsw}_{train,test}.pkl and data/{ds}_meta.json. Constraint logic is NOT applied here:
defective real records (e.g. negative durations) are deliberately kept so that the real-record audit can see them."""
import sys, json, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
SEED = 20261007
ds = sys.argv[1]
if ds == 'cic':
    df = pd.read_pickle('data/cic_raw.pkl')
    n0 = len(df)
    df = df.drop(columns=['Fwd Header Length.1', 'Destination Port', 'source_file'])   # duplicate column; endpoint identifier
    const = [c for c in df.columns if c != 'Label' and df[c].nunique() <= 1]
    df = df.drop(columns=const)
    num = [c for c in df.columns if c != 'Label']
    bad = df[num].isna().any(axis=1) | np.isinf(df[num].to_numpy(dtype='float64')).any(axis=1)
    df = df[~bad]; n1 = len(df)
    df = df.drop_duplicates(); n2 = len(df)
    df['attack_cat'] = df['Label'].where(df['Label'] != 'BENIGN', 'Benign')
    df['y'] = (df['Label'] != 'BENIGN').astype('int8'); df = df.drop(columns=['Label'])
    cat = []
    info = dict(n_raw=n0, n_after_nan_inf=n1, n_after_dedup=n2, constant_dropped=const)
else:
    df = pd.read_pickle('data/unsw_raw.pkl'); df.columns = [c.replace(' ', '') for c in df.columns]
    n0 = len(df)
    df['ct_ftp_cmd'] = pd.to_numeric(df['ct_ftp_cmd'].astype(str).str.strip().replace('', np.nan), errors='coerce')
    for c in ['ct_ftp_cmd', 'is_ftp_login', 'ct_flw_http_mthd']: df[c] = df[c].fillna(0)   # blank = not applicable (documented convention)
    df['attack_cat'] = df['attack_cat'].astype('object').where(df['attack_cat'].notna(), 'Benign').astype(str).str.strip().replace({'Backdoors': 'Backdoor'})
    df = df.drop(columns=['srcip', 'sport', 'dstip', 'dsport', 'Stime', 'Ltime', 'source_file'])  # endpoint identifiers and absolute timestamps
    df = df.rename(columns={'Label': 'y'}); df['y'] = df['y'].astype('int8')
    n1 = len(df); df = df.drop_duplicates(); n2 = len(df)
    cat = ['proto', 'state', 'service']
    for c in cat: df[c] = df[c].astype(str).str.strip()
    info = dict(n_raw=n0, n_after_nan_inf=n1, n_after_dedup=n2, constant_dropped=[])
strat = df['attack_cat'].copy()
vc = strat.value_counts(); rare = vc[vc < 4].index
strat = strat.where(~strat.isin(rare), 'rare')
tr, te = train_test_split(df, test_size=0.30, random_state=SEED, stratify=strat)
tr = tr.reset_index(drop=True); te = te.reset_index(drop=True)
tr.to_pickle(f'data/{ds}_train.pkl'); te.to_pickle(f'data/{ds}_test.pkl')
info.update(seed=SEED, n_train=len(tr), n_test=len(te), features=[c for c in df.columns if c not in ('y', 'attack_cat')], categorical=cat,
            train_classes=tr['attack_cat'].value_counts().to_dict(), test_classes=te['attack_cat'].value_counts().to_dict())
json.dump(info, open(f'data/{ds}_meta.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in info.items() if k != 'features'}, indent=1))
