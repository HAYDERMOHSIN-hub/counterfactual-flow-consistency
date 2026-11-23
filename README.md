# Do Counterfactual Explanations of Network Intrusion Detectors Respect Flow-Feature Consistency? A Constraint-Based Audit

Code, frozen constraint lists, and the re-analysis arrays for the paper
*"Do Counterfactual Explanations of Network Intrusion Detectors Respect Flow-Feature Consistency?
A Constraint-Based Audit"*.

**Paper status:** submitted to the *International Journal of Intelligent Engineering and Systems*
(IJIES), manuscript **20267767**. Not accepted or published — no publication data is claimed here.

The study audits whether counterfactual explanations of flow-based intrusion detectors respect the
eligible inter-feature consistency relations of CIC-IDS2017 and UNSW-NB15, across three tree-ensemble
detectors (RF, XGBoost, HistGB) and three generators: a type-aware unconstrained generator (A-typed),
the same generator with an enforcement component (B), and the nearest real benign flow (C).

## Repository contents
- `code/` — the pipeline: data preparation and split, model training, the generator and constraint
  machinery, the Stage-3 experiment scripts, and the read-only re-analysis scripts
  (`verify_tables_3_5.py`, `verify_R1_6_distance.py`, `exp_R2_1b_identity_sensitivity.py`,
  `verify_R2_3_cic_daycluster.py`, `verify_R1_5_metric_definition.py`).
- `protocol/` — the frozen (187/109) and eligible (44/73) constraint lists.
- `evidence/` — the 48 confirmatory + 18 corrected arm-B per-instance arrays for re-analysis, with
  `SHA256SUMS.txt` and a run guide (`evidence/README.md`). See **Access to the arrays** below.
- `requirements.txt` / `requirements-dice.txt` — core environment; DiCE (R2.2 only) kept separate.
- `REPRODUCIBILITY.md` — the result→script→inputs→source→command→output table and the three tiers.

## Not included (data and results)
- The CIC-IDS2017 and UNSW-NB15 **raw datasets** — obtain from the Canadian Institute for Cybersecurity
  and UNSW Canberra under their own terms (not redistributed here).
- **Trained detector models** (`.pkl`) and the **R1 generation-experiment per-instance arrays**
  (DiCE R2.2, pool-size R2.4, TTL-fixed R2.5, UNSW temporal R2.3) — large and regenerable; their
  per-cell summaries ship with the review package, and the full arrays are available on request.

## Access to the arrays
The confirmatory and corrected arm-B arrays needed to re-analyse the array-based results are included
directly in this repository under `evidence/arrays/` (bit-identical to the originals; verify with
`evidence/SHA256SUMS.txt`). They are **also** deposited on Zenodo as the results record
(DOI 10.5281/zenodo.24227377), per the paper's Data Availability statement. No GitHub Release has been
created yet, so no release-asset link is given here.

## Quick start — re-analyse from a clean checkout
```bash
pip install -r requirements.txt
python code/verify_tables_3_5.py               # Tables 3-5 five-state counts + M1/M2/M3 per cell/arm
python code/verify_R1_6_distance.py            # R1.6 distances, rank-biserial, zero-distance counts
python code/exp_R2_1b_identity_sensitivity.py  # R2.1 identity-sensitivity (all 18 cell x seed)
python code/verify_R2_3_cic_daycluster.py      # R2.3 CIC day-clustered bootstrap CIs
python code/verify_R1_5_metric_definition.py   # R1.5 metric-definition check
```
These read `evidence/arrays/` by default. Expected (reproduced in a clean checkout): Tables 3-5 e.g.
CIC-RF A_typed 1.000/0.380/0.620 and arm B 1.000/0.000/1.000, UNSW arm-C coverage 0.722; UNSW-HGB A-vs-B
rank-biserial +0.18 (p = 0.0011); arm-B zero-distance count 0 in every cell; CIC day-clustered CIs
RF [0.196, 0.492], XGB [0.037, 0.139], HGB [0.072, 0.201]; R1.5 "PASS".

## Re-analyse vs re-generate vs re-train
- **Re-analyse** (above): recompute from the saved arrays in `evidence/` — no models, no raw data.
- **Re-generate**: produce new counterfactuals — needs the trained models (and `dice-ml` for R2.2:
  `pip install -r requirements.txt -r requirements-dice.txt`). Not runnable from this repo alone.
- **Re-train**: rebuild detectors from the raw datasets. Not runnable from this repo alone.
See `REPRODUCIBILITY.md` for the exact per-step commands and inputs.

## Scope of reproducibility
The confirmatory pipeline is seed-fixed (SEED = 20261007) and the re-analysis reproduces the reported
numbers exactly. "Deterministic" applies to that pipeline, **not** to every experiment. On the R2.2
randomness (from `code/exp_R2_2_dice.py`), three things are distinct:
- **Case-sample selection** is seeded: the 200 cases per cell are a deterministic subsample drawn with
  `numpy` seeded by `SEED` (20261007).
- **DiCE-random** method is seeded in the code (`random_seed = 0`).
- **DiCE-genetic** — the variant the R2.2 audit uses — is **not** seed-fixed (no `random_seed` is passed),
  so its counterfactuals are not exactly reproducible run to run.

Also `scipy` was not version-pinned in the original run (the re-analysis here was checked with
scipy 1.18.1). Caveat: these seed settings describe the **current code**; this repository carries no run
log tying this exact code version to the historical runs, so the presence of a seed here is not by itself
proof of how a past run was executed. For these reasons the repository is not described as "fully pinned"
or "exactly reproducible" as a whole.

## Licensing (dual license, by component)
This repository is dual-licensed; see `LICENSE` (overview), `LICENSE-MIT`, and `LICENSE-CC-BY-4.0`.
- **Code** (`code/*.py`): **MIT** (`LICENSE-MIT`).
- **Documentation, constraint lists, and evidence arrays**: **CC BY 4.0** (`LICENSE-CC-BY-4.0`) —
  i.e. `README.md`, `REPRODUCIBILITY.md`, `evidence/README.md`; `protocol/*.json` and
  `code/register_extract.json` (constraint lists / register); and `evidence/arrays/**`.

CC BY 4.0 is used for the non-code materials to **match the verified Zenodo deposit**: the same
confirmatory and corrected arm-B arrays are already deposited, irrevocably, under **CC BY 4.0** on Zenodo
(results record, DOI 10.5281/zenodo.24227377; license verified from the record page), so the same files
are not offered under two different licenses. **Not licensed here:** the CIC-IDS2017 and UNSW-NB15 **raw
datasets** (property of their providers, under their own terms) and any **third-party components**.

**Conditions / open points before publishing (read `LICENSE`):**
1. The evidence arrays are **derived** from CIC-IDS2017 and UNSW-NB15; excluding the raw data does not by
   itself settle derivative rights. The authors are responsible for ensuring the source datasets' terms
   permit licensing these derived arrays — as already asserted by depositing them under CC BY 4.0 on Zenodo.
2. The prior Zenodo deposit (CC BY 4.0) is **irrevocable** and stands on its own; this repository is
   **aligned with it**, not overriding it.
3. Publication is **pending confirmation by all rights-holders**. The licensing choice here was made by the
   submitting author and does **not** by itself evidence the other co-rights-holders' approval. **Do not
   publish this repository publicly until that confirmation is given.** No supervisor approval is asserted.

## Citation
Please cite the paper (IJIES, manuscript 20267767) once its publication details are available, and the
deposited protocol: Zenodo 10.5281/zenodo.23483090 (protocol), 10.5281/zenodo.24227377 (results record).
Do not cite publication venue/volume/pages until the paper is accepted and those are assigned.
