#!/usr/bin/env python3
"""extract_conf_idx.py - reconstruct the confirmatory-sample test-split indices
(data/{ds}_conf_idx_20261008.npy) from the saved per-instance arrays.

Why this exists: the Stage-3 experiment setup (expbase.conf_cases) selects the 1,000 confirmatory
cases per dataset at data/{ds}_conf_idx_20261008.npy (the seed-20261008 draw). That index file is a
run-time input, not one of the deposited evidence arrays, and the training script writes a DIFFERENT
file (data/{ds}_orig_idx.npy, the EXPLORATORY/pilot draw) that must NOT be substituted for it. Every
deposited seed-10 array already records its own `test_idx`, so the confirmatory index can be
recovered exactly from the arrays.

What it does (this is an extraction + an index-level consistency check, NOT a full
generation-readiness check):
  - reads `test_idx` from every seed-10 array of a dataset (confirmatory + corrected-B);
  - verifies they are byte-for-byte identical AND in the same order (same sample, same ordering);
  - writes data/{ds}_conf_idx_20261008.npy.

It does NOT verify feature ordering, label correctness against a re-prepared test split, or that the
models/data needed to REGENERATE counterfactuals are present; those require inputs outside this
repository (raw data, trained models) and are reported as not checked here.

Exit codes: 0 = indices extracted and consistent; non-zero = missing arrays or an inconsistency
(no silent partial success).
Usage: python code/extract_conf_idx.py [confirmatory_dir] [corrected_B_dir] [out_dir]
"""
import os
import sys
import glob
import numpy as np

CONF = sys.argv[1] if len(sys.argv) > 1 else 'evidence/arrays/confirmatory'
CORR_B = sys.argv[2] if len(sys.argv) > 2 else 'evidence/arrays/B_corrected'
OUTDIR = sys.argv[3] if len(sys.argv) > 3 else 'data'
DATASETS = ('cic', 'unsw')
SEED = 10
STAMP = '20261008'

rc = 0
os.makedirs(OUTDIR, exist_ok=True)
for ds in DATASETS:
    files = sorted(glob.glob(os.path.join(CONF, f'{ds}_*_seed{SEED}.npz')) +
                   glob.glob(os.path.join(CORR_B, f'{ds}_*_seed{SEED}.npz')))
    if not files:
        sys.stderr.write(f'ERROR[{ds}]: no seed-{SEED} arrays found under {CONF} / {CORR_B}\n')
        rc = 2
        continue
    ref_name, ref_idx = None, None
    consistent = True
    for f in files:
        idx = np.asarray(np.load(f, allow_pickle=True)['test_idx'])
        if ref_idx is None:
            ref_name, ref_idx = os.path.basename(f), idx
        elif not np.array_equal(ref_idx, idx):
            same_set = set(ref_idx.tolist()) == set(idx.tolist())
            sys.stderr.write(
                f'ERROR[{ds}]: test_idx in {os.path.basename(f)} differs from {ref_name} '
                f'(same set but different order: {same_set}); the confirmatory sample is not '
                f'consistent across arrays - refusing to write an index.\n')
            consistent = False
            rc = 3
    if not consistent:
        continue
    out = os.path.join(OUTDIR, f'{ds}_conf_idx_{STAMP}.npy')
    np.save(out, ref_idx)
    print(f'[{ds}] {len(files)} seed-{SEED} arrays agree on test_idx (n={len(ref_idx)}); '
          f'wrote {out}  first5={ref_idx[:5].tolist()}')

if rc == 0:
    print('\nOK: confirmatory indices extracted and consistent across all arrays. '
          'Note: this is an index-level check only; feature ordering and regeneration readiness '
          'are NOT asserted (they need the raw data and trained models, which are not in this repo).')
else:
    sys.stderr.write('\nFAILED: see errors above.\n')
sys.exit(rc)
