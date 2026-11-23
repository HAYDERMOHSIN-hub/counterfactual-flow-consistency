"""make_manifest.py - write a SHA-256 manifest of the whole reproducibility package (R1.3 / ED.3).
Hashes everything under code/, protocol/, data/ (models + meta + indices), results/, experiments/ and
the key logs, with sizes, sorted by path. Deterministic; writes MANIFEST_SHA256.txt at the package root.
Usage: python code\\make_manifest.py
"""
import hashlib, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cfg

ROOT = cfg.ROOT
INCLUDE = ['code', 'protocol', 'data', 'results', 'experiments']
LOGS = ['validation_report.json', 'reproduce_gate.json', 'check_env_report.json', 'stage3.log', 'stage3_dice.log']
SKIP_EXT = ('.pyc',)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


rows = []
for sub in INCLUDE:
    base = os.path.join(ROOT, sub)
    if not os.path.isdir(base):
        continue
    for root, _, files in os.walk(base):
        if '__pycache__' in root:
            continue
        for fn in sorted(files):
            if fn.endswith(SKIP_EXT):
                continue
            p = os.path.join(root, fn)
            rows.append((os.path.relpath(p, ROOT).replace('\\', '/'), sha256(p), os.path.getsize(p)))
for lg in LOGS:
    p = os.path.join(cfg.LOGS, lg)
    if os.path.exists(p):
        rows.append((os.path.relpath(p, ROOT).replace('\\', '/'), sha256(p), os.path.getsize(p)))
rows.sort(key=lambda r: r[0])

out = os.path.join(ROOT, 'MANIFEST_SHA256.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('# SHA-256 manifest - IJIES 20267767 reproducibility package\n')
    f.write(f'# generated {time.strftime("%Y-%m-%d %H:%M:%S")}  SEED=20261007\n')
    f.write('# libraries: Python 3.13, numpy 2.5.3, pandas 3.0.5, scikit-learn 1.9.1, xgboost 2.1.4\n')
    f.write(f'# files: {len(rows)}\n')
    for rel, h, sz in rows:
        f.write(f'{h}  {rel}  ({sz} bytes)\n')
print(f'Wrote {out} with {len(rows)} files.')
