"""exp3_arms_rerun.py - STAGE 3A (generation): re-run the confirmatory arms on the REBUILT models
for one (dataset, detector), so the fair-comparison baseline and the new experiments share the same
rebuilt detectors. Runs the identical 8-run confirmatory schedule:
    A_typed seeds 10,11,12 ; B seeds 10,11,12 ; A(raw) seed 10 ; C seed 10
Arms A_typed/A/C are byte-for-byte unchanged by the 2 Nov correction, so they must reproduce the
saved original_48 arrays; arm B is the corrected arm and must reproduce the saved corrected_B_18
arrays. The gate (exp3_reproduce_gate.py) checks that. Also records the per-instance model-query
count (R1.4). Checkpoint: skips any arm whose output already exists.
Usage: python exp3_arms_rerun.py <cic|unsw> <RF|XGB|HGB>
"""
import os, sys, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace'); sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
import expbase as EB

ds, det = sys.argv[1], sys.argv[2]
OUT = os.path.join(EB.EXP, 'repro', 'raw'); os.makedirs(OUT, exist_ok=True)
sp, feats, meta = EB.build_space(ds)
model = EB.load_model(ds, det)
X0, c0, idx = EB.conf_cases(ds, feats)

RUNS = [('A_typed', 10), ('B', 10), ('C', 10), ('A', 10),
        ('A_typed', 11), ('B', 11), ('A_typed', 12), ('B', 12)]
stop = False
for arm, seed in RUNS:
    fn = os.path.join(OUT, f'{ds}_{det}_{arm}_seed{seed}.npz')
    if os.path.exists(fn):
        print(f'skip {os.path.basename(fn)} (exists)', flush=True); continue
    t = time.time()
    r = EB.run_arm(sp, model, X0, arm, seed)
    info = EB.save_arm(fn, sp, feats, arm, seed, X0, c0, idx, r)
    flag = '' if info['n_mis'] == 0 else f"  !! IMMUTABLE MISMATCH n={info['n_mis']}"
    print(f"{ds} {det} {arm} s{seed}: flip {info['flip']:.3f} queries/inst {info['queries']} "
          f"immMis {info['n_mis']} [{time.time()-t:.0f}s]{flag}", flush=True)
    if info['n_mis'] > 0:
        stop = True
print('IMMUTABILITY-STOP' if stop else f'cell {ds}_{det} OK', flush=True)
