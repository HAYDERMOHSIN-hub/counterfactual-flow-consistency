# Evidence: arrays for re-analysis (no regeneration needed)

This folder lets you RE-ANALYSE the array-based results directly, without retraining models or
regenerating counterfactuals.

## Contents
- `arrays/confirmatory/` — the 48 confirmatory per-instance arrays (seed 10; arms A_typed, A (raw),
  B (initial full-list), C) for the six dataset×detector cells. These are the ORIGINAL confirmatory arrays.
- `arrays/B_corrected/` — the 18 CORRECTED arm-B arrays (eligible-list enforcement re-run; manuscript
  Section 3.5, deviation 1). The manuscript's arm-B distances come from these, not from
  `arrays/confirmatory/*_B_*`.
- `SHA256SUMS.txt` — SHA-256 of every array here (bit-identical to the deposited originals; not
  recompressed or modified). Verify with `sha256sum -c SHA256SUMS.txt` from this folder.

These arrays are also deposited on Zenodo as the results record (DOI 10.5281/zenodo.24227377), per the
paper's Data Availability statement.

## Scope — what these support, and what they do NOT
RE-ANALYSABLE from these arrays, each by a runnable script that recomputes from the arrays (never copying
from the tables or any summary JSON):
- the confirmatory five-state counts and M1/M2/M3 of **Tables 3–5** (`verify_tables_3_5.py`);
- the **R1.6** paired distances / rank-biserial / zero-distance counts (`verify_R1_6_distance.py`);
- the **R2.1** identity-sensitivity secondary endpoint, all 18 cell×seed combinations
  (`exp_R2_1b_identity_sensitivity.py`);
- the **R2.3** CIC day-clustered bootstrap (`verify_R2_3_cic_daycluster.py`);
- the **R1.5** metric-definition check (`verify_R1_5_metric_definition.py`).

NOT recomputable from this folder (summaries only): the R1 generation experiments — DiCE (R2.2),
pool-size sweep (R2.4), TTL-fixed (R2.5) and the UNSW temporal split (R2.3 temporal). Their per-cell
summaries ship with the review package; their per-instance arrays are NOT here (large, regenerable).
Re-generating any of them needs the trained detector models and, for R2.2, `dice-ml`
(see `../requirements-dice.txt`); re-training the detectors needs the raw CIC-IDS2017 / UNSW-NB15
datasets from their providers. (The historical `analyze_confirmatory_corrected_ORIGINAL.py` is kept as a
reference to the original analysis; it depends on result directories not in this repository and does not
run from the package alone — `verify_tables_3_5.py` is the runnable re-analysis interface instead.)

## Run the re-analyses (from the repository root, after `pip install -r requirements.txt`)
```bash
# Tables 3–5 — five-state counts + M1/M2/M3 per cell and arm (arm B from the corrected set)
python code/verify_tables_3_5.py
# R1.6 — paired distances, rank-biserial, zero-distance counts
python code/verify_R1_6_distance.py
# R2.1 — identity-sensitivity secondary endpoint (all 18 cell×seed combinations)
python code/exp_R2_1b_identity_sensitivity.py
# R2.3 — CIC day-clustered bootstrap CIs
python code/verify_R2_3_cic_daycluster.py
# R1.5 — metric-definition check
python code/verify_R1_5_metric_definition.py
```
Expected (reproduced in a clean checkout): Tables 3–5 — e.g. CIC-RF A_typed M1/M2/M3 = 1.000/0.380/0.620,
arm B 1.000/0.000/1.000, UNSW arm-C coverage 0.722; UNSW-HGB A-vs-B rank-biserial +0.18 (p = 0.0011);
arm-B exactly-zero-distance count = 0 in every cell; CIC day-clustered CIs RF [0.196, 0.492],
XGB [0.037, 0.139], HGB [0.072, 0.201]; R2.1 CIC-RF M2 0.380→0.034 (excl. identities); R1.5 "PASS".
