# Reproducibility

Three tiers, kept distinct:

- **RE-ANALYSE** — recompute results from the saved arrays in `evidence/`. No models, no new
  counterfactuals, no raw data. This is the part that runs from a clean checkout of this repository.
- **RE-GENERATE** — produce new counterfactuals (needs trained detector models; `dice-ml` for R2.2).
- **RE-TRAIN** — rebuild the detectors from the raw CIC-IDS2017 / UNSW-NB15 datasets.

Pinned where recorded: Python 3.13; numpy 2.5.3, pandas 3.0.5, scikit-learn 1.9.1, xgboost 2.1.4
(from the original run environment; see `requirements.txt`). `scipy` was not pinned in the original
run and is left unpinned (the re-analysis here was checked with scipy 1.18.1, a separate verification
environment). `dice-ml` 0.12 is in `requirements-dice.txt` and is needed only to re-generate R2.2.
SEED = 20261007. This repository is **not** "fully pinned / exactly reproducible": scipy is unpinned, and
in R2.2 the DiCE-**genetic** variant (the one the audit uses) passes no `random_seed`, while DiCE-random
is seeded (`random_seed = 0`) and the case-sample selection is seeded by SEED. The seed settings describe
the current code, not a logged historical run (see README, "Scope of reproducibility").

## What is and is not in this repository
- IN: `code/`, `protocol/`, and `evidence/arrays/` (the 48 confirmatory + 18 corrected arm-B arrays,
  bit-identical to the deposited originals; also on Zenodo 10.5281/zenodo.24227377).
- NOT IN (regenerable / external): the raw datasets (from the Canadian Institute for Cybersecurity and
  UNSW Canberra), the trained detector models (`.pkl`), and the R1 generation-experiment per-instance
  arrays (DiCE/pool-size/TTL/temporal — large; their summaries ship with the review package; the full
  arrays are available on request). A full manifest over models+arrays is produced on the author's
  machine and is not part of this repository.

## RE-ANALYSE — runs from a clean checkout (from the repository root)
```bash
pip install -r requirements.txt
python code/verify_tables_3_5.py                 # Tables 3-5: five-state counts + M1/M2/M3 per cell/arm
python code/verify_R1_6_distance.py              # R1.6: paired distances, rank-biserial, zero-distance counts
python code/exp_R2_1b_identity_sensitivity.py    # R2.1: identity-sensitivity (all 18 cell x seed)
python code/verify_R2_3_cic_daycluster.py        # R2.3: CIC day-clustered bootstrap CIs
python code/verify_R1_5_metric_definition.py     # R1.5: metric-definition check
```
These default to the arrays in `evidence/arrays/`; see `evidence/README.md` for expected outputs.

| Result | Script (tier) | Inputs | Where to get the inputs | Command (from repo root) | Outputs |
|---|---|---|---|---|---|
| R1.6 distances / rank-biserial / zero-dist | `verify_R1_6_distance.py` (re-analyse) | confirmatory A_typed+C arrays, corrected B arrays | `evidence/arrays/` (in repo) | `python code/verify_R1_6_distance.py` | `R1_6_distance_verified.json` |
| R2.3 CIC day-clustered CIs | `verify_R2_3_cic_daycluster.py` (re-analyse) | confirmatory A_typed arrays | `evidence/arrays/confirmatory/` (in repo) | `python code/verify_R2_3_cic_daycluster.py` | `R2_3_cic_daycluster_verified.json` |
| R1.5 metric definition | `verify_R1_5_metric_definition.py` (re-analyse) | none (synthetic cases) | — | `python code/verify_R1_5_metric_definition.py` | prints PASS |
| **Tables 3–5 (M1/M2/M3, five states)** | `verify_tables_3_5.py` (re-analyse) | confirmatory A_typed/A/C + corrected B arrays | `evidence/arrays/` (in repo) | `python code/verify_tables_3_5.py` | `tables_3_5_verified.json` |
| Tables 3–5 (REGENERATE the arrays) | `exp3_arms_rerun.py <ds> <det>` then `exp3_reproduce_gate.py` (re-generate / verify) | trained models + confirmatory index | NOT in repo (re-train, or author on request) | `python code/exp3_arms_rerun.py cic RF` … | per-instance arrays + gate log |
| R2.1 identity sensitivity | `exp_R2_1b_identity_sensitivity.py` (re-analyse) | A_typed arrays, 6 cells × seeds 10/11/12 | `evidence/arrays/confirmatory/` (in repo, default path) | `python code/exp_R2_1b_identity_sensitivity.py` | `R2_1b_identity_sensitivity.json` |
| R2.2 DiCE audit | `exp_R2_2_dice.py <ds> <det>` (re-generate) | models + index + `dice-ml` | NOT in repo | `pip install -r requirements-dice.txt; python code/exp_R2_2_dice.py cic RF` … | DiCE arrays + summaries |
| R2.4 pool-size sweep | `exp_R2_4_poolsize.py <ds> <det>` (re-generate) | models + index | NOT in repo | `python code/exp_R2_4_poolsize.py cic RF` … | pool-size summaries |
| R2.5 TTL-fixed | `exp_R2_5_ttl_fixed.py <det>` (re-generate) | models + index | NOT in repo | `python code/exp_R2_5_ttl_fixed.py RF` | TTL summaries |
| R2.3 temporal split | `exp_R2_3_temporal.py` (re-generate) | raw UNSW-NB15 | NOT in repo | `python code/exp_R2_3_temporal.py` | temporal arrays + summaries |
| Table 2 (detectors) | `04_train_models.py <ds>` (re-train) | prepared data | NOT in repo | `python code/04_train_models.py cic` | detector performance + models |

Rows marked "NOT in repo" cannot be run from this repository alone; they need the models, raw data, or
`dice-ml` as noted. No row is claimed to run from the repo unless its inputs are in `evidence/`.
