"""01_prepare_data.py - Load raw CIC-IDS2017 (MachineLearningCVE) and UNSW-NB15 (four raw parts),
apply minimal, documented cleaning, and write compact pickles. No constraint logic lives here."""
import sys, glob, os, numpy as np, pandas as pd
RAW = sys.argv[1]; OUT = sys.argv[2]

def cic():
    fs = sorted(glob.glob(os.path.join(RAW, 'CIC-IDS2017', '*.csv')))
    parts = []
    for f in fs:
        d = pd.read_csv(f, encoding='latin1', low_memory=False)
        d.columns = [c.strip() for c in d.columns]
        d['source_file'] = os.path.basename(f).split('.pcap')[0]
        parts.append(d); print(os.path.basename(f), d.shape, flush=True)
    df = pd.concat(parts, ignore_index=True); del parts
    df['Label'] = df['Label'].astype(str).str.strip().str.replace('\x96', '-', regex=False).str.replace('ï¿½', '-', regex=False)
    print(df['Label'].value_counts())
    df.to_pickle(os.path.join(OUT, 'cic_raw.pkl'))

def unsw():
    feat = pd.read_csv(os.path.join(RAW, 'UNSW-nb15 Dataset', 'UNSW-NB15 Dataset CSV files', 'NUSW-NB15_features.csv'), encoding='latin1')
    names = [n.strip() for n in feat['Name']]
    parts = []
    for i in range(1, 5):
        f = os.path.join(RAW, 'UNSW-nb15 Dataset', 'UNSW-NB15 Dataset CSV files', f'UNSW-NB15_{i}.csv')
        d = pd.read_csv(f, header=None, names=names, low_memory=False, encoding='latin1')
        d['source_file'] = f'UNSW-NB15_{i}'
        parts.append(d); print(f, d.shape, flush=True)
    df = pd.concat(parts, ignore_index=True)
    print(df.dtypes.to_string()); print(df['Label'].value_counts()); print(df['attack_cat'].value_counts(dropna=False))
    df.to_pickle(os.path.join(OUT, 'unsw_raw.pkl'))

{'cic': cic, 'unsw': unsw}[sys.argv[3]]()
