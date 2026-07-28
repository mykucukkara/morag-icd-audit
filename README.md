# Audit and verification code for "Why a Retrieval-Augmented LLM Loses to TF-IDF at ICD-10 Coding"

This repository holds the audit protocol from the paper, not the pipeline it audits. The point of
the paper is the protocol: a way to check whether an evidence-grounded LLM coding system actually
does what it reports. Everything needed to apply that protocol to another system, and to re-derive
every number the paper reports from stored model outputs, is here.

The pipeline implementation is available from the author on request.

## What the protocol does

1. **A note-blind floor.** A constant predictor that emits the K most frequent training codes and
   never reads the note. Any system that does not clear it extracts no per-note signal. Costs
   nothing, and every LLM arm in our ladder fails it (`scripts/40`).
2. **A loss decomposition.** An oracle-over-shortlist bound and a random-pruning null, both at the
   system's realized budget, separating what retrieval makes available from what the selector does
   with it (`scripts/40`).
3. **Note-level paired significance testing.** Paired bootstrap intervals plus an
   approximate-randomization test, resampled at the note level because code decisions within a note
   are not independent (`scripts/37`).
4. **An evidence-provenance measurement.** What fraction of the "supporting evidence" a system
   displays comes from a different patient, with the chance baseline the design implies
   (`scripts/38`).
5. **A grounding metric the model cannot self-report.** Whether the quoted evidence string actually
   occurs in the passage the scorer was shown (`src/morag_icd/evaluation/evidence_metrics.py`).
6. **A provenance guard.** Refuses to build manuscript tables from fixture or underpowered data
   (`src/morag_icd/reporting/provenance_guard.py`).
7. **A number-level verification pass.** Re-reads every artifact and compares it against the value
   the manuscript states, at the precision the manuscript uses, failing on any disagreement
   (`scripts/44`).

Item 7 is the one we would encourage others to copy. On its first run it caught two defects in our
own reporting: a value rounded up from 0.186 to 0.187 in four places, and a sentence implying a
difference between two numbers that were identical.

## Reproducing the paper's numbers

    python scripts/44_verify_manuscript_numbers.py \
        --artifacts tables/generated/artifacts/primary_campaign \
        --scalability tables/generated/artifacts/scalability_v2/metrics \
        --sections . --splits-root ./none

Expected: `54 ok, 0 mismatched, 0 skipped`.

Nine further checks in the paper's own run derive from the MIMIC-IV split files (test-set sizes,
gold codes per note, and the recall ceiling at a fifteen-code budget). Those files cannot be
redistributed, so those checks are skipped here; `data_summaries/` carries the aggregate counts
they produce. With MIMIC-IV access, point `--splits-root` at your split tree and the run reports
63 checks.

To regenerate the tables:

    python scripts/42_generate_revision_tables.py \
        --artifacts tables/generated/artifacts/primary_campaign \
        --scalability-metrics tables/generated/artifacts/scalability_v2/metrics \
        --outdir tables/generated/tables/top50

## What is deliberately absent

- **No data.** No clinical text, no subject or admission identifiers, no prediction files. MIMIC-IV
  is available to credentialed researchers through PhysioNet under its data use agreement and is
  not redistributed here. The stored artifacts in `tables/generated/artifacts/` are aggregates —
  metrics, contrasts, confidence intervals — and were scanned for identifiers before release.
- **No credentials or infrastructure.** Cluster hostnames, account names, absolute paths and job
  scripts were removed. Model paths in `configs/models*.yaml` are `${MODELS_ROOT}` placeholders.
- **No pipeline.** `src/` contains only the evaluation and reporting modules the audit scripts
  import.

## Layout

    scripts/         the audit, analysis, verification and build scripts
    src/morag_icd/   evaluation metrics and the provenance guard
    figures/         the five figure generators and their captions
    tables/          generated tables and the aggregate artifacts they read
    configs/         run configurations referenced in the Methods section
    data_summaries/  split sizes and label-space counts (no identifiers)

## Citation

Küçükkara, M. Y. (2026). Why a Retrieval-Augmented LLM Loses to TF-IDF at ICD-10 Coding: A
Component-Wise Cautionary Study. *Manuscript under review.*

## License

MIT (see `LICENSE`). MIMIC-IV itself is governed by the PhysioNet Credentialed Health Data Use
Agreement and is not covered by this license.
