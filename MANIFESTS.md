# Integrity manifests — scope, sources, and how to verify

This repository ships **three independent SHA-256 manifests**. Each one covers **only the
files listed inside it** — no manifest covers another manifest's files, and no manifest
lists itself. Read this table before quoting any manifest's scope.

| Reference material | Version / source | Manifest file (in this repo) | What it covers | Verify command (from the repo root) |
|---|---|---|---|---|
| Evidence arrays — 48 original confirmatory + 18 corrected arm-B (`.npz`) | this repo `evidence/arrays/`; also deposited on Zenodo results record 10.5281/zenodo.24227377 | `evidence/SHA256SUMS.txt` | the **66 arrays only** (not code, not constraint lists) | `cd evidence && sha256sum -c SHA256SUMS.txt` |
| Analysis/verification code + constraint lists | this repo `code/` and `protocol/` | `SOURCE_PROTOCOL_SHA256SUMS.txt` (repo root) | the **24 code files** (`code/**`, incl. the constraint register `code/register_extract.json`) **and the 4 frozen/eligible constraint lists** (`protocol/**`) — nothing else | `git -c core.autocrlf=false clone <url>` then `sha256sum -c SOURCE_PROTOCOL_SHA256SUMS.txt` (on Linux the default already preserves LF; these files are stored with LF endings) |
| R1 revision run outputs (logs, verification outputs, per-cell summaries) | this repo `review_evidence/R1/` | `review_evidence/R1/SHA256SUMS.txt` | the **38 run-output files** under `review_evidence/R1/{logs,verification,experiment_summaries}` — not the README, FILE_LIST or the manifest itself | `cd review_evidence/R1 && sha256sum -c SHA256SUMS.txt` (these files are marked `-text` in `.gitattributes`, so they verify byte-for-byte on every platform) |

## Material that is NOT in this repository (referenced, not re-hashed here)
- **Pre-specified protocol v1.0 and methodological addendum** — deposited on Zenodo:
  concept DOI 10.5281/zenodo.23483089; protocol (v1.0) 10.5281/zenodo.23483090;
  methodological addendum 10.5281/zenodo.23672368. The protocol deposit records its own
  hash values for the protocol, seeds and excluded-identifier lists, and Zenodo publishes a
  per-file checksum for every deposited file. Those published checksums — at their stated
  deposit dates — are the record of what was fixed before the confirmatory draw. **This
  repository does not re-derive those hashes and does not restate them with a new date.**
- **Trained detector models (`.pkl`) and the R1 generation-experiment per-instance `.npz`
  arrays** — regenerable by the released scripts; available from the authors on request.
  A full manifest over models+arrays is produced on the author's machine and is not part of
  this repository.

## Integrity is not chronology
A SHA-256 digest proves only that a **file's bytes are intact and unchanged** — it is
evidence of *integrity*, not of *when the file was created or when a computation was run*.
The manifests in this repository (and their git commit times) therefore establish that the
committed files match their recorded digests; they do **not**, on their own, establish the
execution date of any experiment. Chronology for the pre-specified protocol rests on the
dated Zenodo deposit above, not on these repository manifests.
