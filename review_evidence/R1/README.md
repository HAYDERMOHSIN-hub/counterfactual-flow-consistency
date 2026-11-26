# review_evidence/R1 — R1 revision run outputs (IJIES 20267767)

This folder collects files produced during the R1 revision of the manuscript "Do
Counterfactual Explanations of Network Intrusion Detectors Respect Flow-Feature
Consistency? A Constraint-Based Audit". They are provided so that numbers quoted in the
Response-to-Reviewers letter and in Appendix G of the Supplementary can be traced to a
concrete file.

## What these files are (and are not)
- **Script-generated run outputs** — the contents of `logs/` (run and validation logs),
  `verification/*_verified.json` and `verification/*_output.txt` (recomputed from the saved
  arrays by the `verify_*` scripts), and `experiment_summaries/*.json` (per-cell summaries).
  These were produced by the released scripts under `code/` during the R1 revision.
- **Hand-written documentation** — `verification/R1_4_query_budget_derivation.md` is a
  written derivation note, not a script output. This `README.md` and `FILE_LIST.txt` were
  written when assembling this folder. `SHA256SUMS.txt` is generated at packaging time.
- **Script copies** — the `*.py` files under `verification/` are copies of the corresponding
  scripts in `code/`, included for convenience.
- No detector was retrained and no counterfactual was regenerated to assemble this folder.
  No raw datasets, trained model binaries, secrets or private correspondence are included.

## Dates — what is and is not established
The retained output files do **not** carry an embedded execution timestamp, so **the exact
execution date of the R1 experiments is not established by these records.** What can be
stated:
- The corrected generator/constraints/utility code modules are dated **2 November 2026** in
  Appendix G.10 — that is a *code-version* date, not the date these outputs were produced.
- The R1 revision experiments (Appendix G) were run **during the revision round**, i.e.
  after the reviewer report; the output files above are their results.
- File modification times, archive-extraction times and git commit times are **not** taken
  as evidence of the experiment execution date.
Where an exact date is not documented, it is left stated as unestablished rather than
guessed.

## Which manifest covers what (do not conflate)
`SHA256SUMS.txt` **in this folder covers only the 38 run-output files listed in it** (the
contents of `logs/`, `verification/` and `experiment_summaries/`). It does **not** cover,
and must not be read as covering, the README, FILE_LIST or itself. The other manifests:
- the **evidence arrays** (`.npz`) are covered by `evidence/SHA256SUMS.txt` at the
  repository root (that manifest covers the 66 arrays **only** — not code, not constraint
  lists);
- the **source code and the frozen/eligible constraint lists** are covered by
  `SOURCE_PROTOCOL_SHA256SUMS.txt` at the repository root.
See `MANIFESTS.md` at the repository root for the full map (material ← source ← manifest ←
verify command) and for the external pre-specified-protocol deposit.

Verify this folder from the repository root with:
`cd review_evidence/R1 && sha256sum -c SHA256SUMS.txt`
(The files here are marked `-text` in `.gitattributes`, so the digests verify byte-for-byte
on every platform.)

## Integrity is not chronology
A SHA-256 digest proves a file's bytes are intact; it does not prove when the file was
created or when a computation was run. These manifests establish file integrity, not
execution dates.

## What each file supports
- `logs/reproduce_gate.json` — per-cell reproduce-gate check (flip-agreement and
  metric-delta tolerances; evaluated-candidates budget 1,260). Supports R1.3/R1.4.
- `verification/R1_4_query_budget_derivation.md` — written derivation of the 1,260
  evaluated candidates per instance vs. the distinct-query count. Supports R1.4.
- `verification/R1_6_distance_verified.json` (+ `R1_6_distance_output.txt`) — paired
  B-vs-C and A-vs-B percentile-distance verification. Supports R1.6 and Appendix G.
- `verification/R2_3_cic_daycluster_verified.json` (+ `.txt`) — CIC-IDS2017 day-clustered
  bootstrap CIs for M2. Supports R2.3.
- `verification/R1_5_output.txt` — metric-definition check. Supports R1.5.
- `experiment_summaries/*.json` — per-cell summaries behind the Appendix G tables
  (R2.2 DiCE, R2.4 benign-pool size, R2.5 TTL-fixed, R2.3 temporal, identity sensitivity).
- `logs/stage3*.log`, `logs/validation_report.json` — generation/validation run logs.

## Relationship to the public deposits
- Code, constraint lists and the 48 original + 18 corrected arm-B arrays:
  this repository and its Zenodo archive (concept DOI 10.5281/zenodo.24541885).
- Original confirmatory arrays and result tables:
  Zenodo results record 10.5281/zenodo.24227377.
- Pre-specified protocol v1.0 and methodological addendum (external):
  Zenodo 10.5281/zenodo.23483089 / 23483090 / 23672368 (see `MANIFESTS.md`).
