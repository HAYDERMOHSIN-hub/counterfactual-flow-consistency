# review_evidence/R1 — R1 revision run outputs (IJIES 20267767)

This folder collects the **run-output files** produced during the R1 revision of the
manuscript "Do Counterfactual Explanations of Network Intrusion Detectors Respect
Flow-Feature Consistency? A Constraint-Based Audit". They are the actual outputs of the
released scripts, generated on 2 November 2026. They are provided so that the numbers
quoted in the Response-to-Reviewers letter and in Appendix G of the Supplementary can be
traced to a concrete output file.

## Provenance and integrity statement
- These are **original run outputs from the R1 analysis**, not values re-typed from the
  manuscript tables and not reconstructed after the fact. No file here was regenerated or
  edited while assembling this folder.
- Every file was produced by a script that is public in this repository under `code/`
  (e.g. `code/exp3_reproduce_gate.py`, `code/verify_R1_6_distance.py`,
  `code/verify_R2_3_cic_daycluster.py`, `code/verify_R1_5_metric_definition.py`,
  `code/exp_R2_2_dice.py`, `code/exp_R2_4_poolsize.py`, `code/exp_R2_5_ttl_fixed.py`).
- No detector was retrained and no counterfactual was regenerated to assemble this folder.
- No raw datasets, trained model binaries, secrets, or private correspondence are included.

## Hash scope (read this before using SHA256SUMS.txt)
`SHA256SUMS.txt` in **this folder** covers **only the run-output files listed in it**
(the contents of `logs/`, `verification/`, and `experiment_summaries/`). It does **not**
cover, and must not be read as covering:
- the **evidence arrays** (`.npz`) — their SHA-256 digests are in `evidence/SHA256SUMS.txt`
  at the repository root;
- the **source code and constraint lists** — their digests are in the same
  `evidence/SHA256SUMS.txt`.
Each manifest covers only the files it lists.

## What each file supports
- `logs/reproduce_gate.json` — the per-cell reproduce-gate check (flip-agreement and
  metric-delta tolerances; evaluated-candidates budget 1,260). Supports R1.3/R1.4.
- `verification/R1_4_query_budget_derivation.md` — derivation of the 1,260 evaluated
  candidates per instance vs. the distinct-query count. Supports R1.4.
- `verification/R1_6_distance_verified.json` (+ `R1_6_distance_output.txt`,
  `verify_R1_6_distance.py`) — paired B-vs-C and A-vs-B percentile-distance verification.
  Supports R1.6 and Appendix G.
- `verification/R2_3_cic_daycluster_verified.json` (+ `.txt`) — CIC-IDS2017 day-clustered
  bootstrap CIs for M2. Supports R2.3.
- `verification/R1_5_output.txt` (+ `verify_R1_5_metric_definition.py`) — metric-definition
  check. Supports R1.5.
- `experiment_summaries/*.json` — per-cell summaries behind the Appendix G tables
  (R2.2 DiCE, R2.4 benign-pool size, R2.5 TTL-fixed, R2.3 temporal, identity sensitivity).
- `logs/stage3*.log`, `logs/validation_report.json` — generation/validation run logs.

## Relationship to the public deposits
- Code, constraint lists, and the 48 original + 18 corrected arm-B arrays:
  this repository and its Zenodo archive (concept DOI 10.5281/zenodo.24541885).
- Original confirmatory arrays and result tables:
  Zenodo results record 10.5281/zenodo.24227377.
