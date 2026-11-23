"""cfg.py - path resolution for the IJIES R1 local-run package.
Run everything from the package root (the .bat files do `cd /d %~dp0`).
RAW data location is read from RAW_PATH.txt in the package root; edit that file if your
raw datasets live elsewhere. Default assumes the package sits inside the «البحث الرابع» folder.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                      # package root
DATA = os.path.join(ROOT, 'data')
RESULTS = os.path.join(ROOT, 'results')
LOGS = os.path.join(ROOT, 'logs')
REF = os.path.join(ROOT, 'reference')
for d in (DATA, RESULTS, LOGS):
    os.makedirs(d, exist_ok=True)

def raw_path():
    p = os.path.join(ROOT, 'RAW_PATH.txt')
    if os.path.exists(p):
        v = open(p, encoding='utf-8').read().strip().strip('"')
        if v:
            return v
    # default: two levels up from package root (…/البحث الرابع/IJIES_.../LocalRun -> البحث الرابع)
    return os.path.dirname(os.path.dirname(ROOT))

SEED = 20261007
DATASETS = ('cic', 'unsw')
DETECTORS = ('RF', 'XGB', 'HGB')
