# Code and audit protocol for "Context and Capacity Interact: Why Single-Factor Evaluations Misjudge RAG-LLM ICD-10 Coding"

This repository holds both the RAG-LLM ICD-10 coding pipeline the paper studies and the audit
protocol it is studied with. The paper is a component-wise autopsy, so withholding the system
would leave the central question unanswerable: a reader could not check whether the result comes
from the design or from a defect in the prompting, parsing, ranking or retrieval code. An
independent reviewer made exactly that objection, and it was right.

The headline finding is that scorer context and model capacity interact. Neither factor helps on
its own, and the same pipeline sits on opposite sides of the note-blind floor depending on where
it is run: starved of note context at 3B it falls significantly below a predictor that never reads
the note, while at 14B with truncated context it clears that floor by 0.130 and comes within 0.024
of tuned TF-IDF. An evaluation varying either factor alone would conclude that neither mattered.

## What the protocol does

1. **A note-blind floor.** A constant predictor that emits the K most frequent training codes and
   never reads the note. Any system that does not clear it extracts no per-note signal. Costs
   nothing, and every LLM arm in our ladder fails it (`scripts/40`).
2. **A loss decomposition at matched cardinality.** An oracle-over-shortlist bound and a random
   null that emit, per note, exactly as many codes as the system did — micro-F1 moves with
   cardinality, so an unmatched null flatters or punishes the system for free (`scripts/47`).
   Paired with stage-wise gold-code retention, which localizes the loss to a component rather than
   apportioning percentages of an F1 gap that is not additive.
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

8. **A complete factorial when two factors are claimed to interact** (`scripts/48`). Our first
   version assembled a 2 x 2 from two different samples; a reviewer pointed out that differences
   taken across samples cannot establish an interaction. The missing cell was re-run and the
   difference-in-differences computed properly.

Item 7 is the one we would encourage others to copy. It caught a value rounded up from 0.186 to
0.187 in four places, and a sentence implying a difference between two numbers that were identical.
It did not catch everything: private annotations printed into the reference list, stale
cross-references and a superseded figure caption all survived it, because it only looked at numbers.
Those checks were added afterwards, which is the honest version of the lesson.

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
- **No credentials or infrastructure.** Cluster hostnames, account names, absolute paths and the
  SLURM job scripts are not part of the release; model paths in `configs/models*.yaml` are
  `${MODELS_ROOT}` placeholders. `scripts/generate_*_campaign.py` shows how the runs were laid out.

## Layout

    scripts/         data preparation, index building, experiment runner, audit and analysis
    src/morag_icd/   the full pipeline: retrieval, LLM scoring, evidence, verification, baselines,
                     optimization, evaluation metrics, provenance guard
    figures/         the five figure generators and their captions
    tables/          generated tables and the aggregate artifacts they read
    configs/         run configurations referenced in the Methods section
    data_summaries/  split sizes and label-space counts (no identifiers)

## Running the pipeline

The pipeline needs MIMIC-IV (PhysioNet credentialed access), a local instruction-tuned LLM and a
local embedding model; nothing is downloaded at run time. In order:

    python scripts/01_preprocess_mimic.py     # discharge summaries -> working corpus
    python scripts/02_build_icd_kb.py         # ICD-10 knowledge base from the code descriptions
    python scripts/03_create_splits.py        # subject-disjoint train/validation/test, Top-N label sets
    python scripts/04_build_bm25_index.py
    python scripts/05_generate_embeddings.py
    python scripts/06_build_faiss_index.py
    python scripts/07_run_experiment.py --experiment-id E14 --seed 42 --top-n 50 --split test ...
    python scripts/10_evaluate_results.py     # metrics over the stored predictions

Model paths live in `configs/models*.yaml` as `${MODELS_ROOT}` placeholders. Runtime behaviour —
candidate count, evidence budget, whether the scorer sees the note — is in `configs/real_*.yaml`;
the difference between `real_batched.yaml` and `real_steelman.yaml` is the `prompt_note_max_chars`
key, which is what supplies the discharge note to the scorer.

The prompts are in `src/morag_icd/llm/prompts.py`, `code_scorer.py` and `contrastive_verifier.py`;
the JSON parser and schema handling, where a missing confidence field once corrupted ranking, are in
`llm/json_parser.py` and `code_scorer.py`. The few-shot examples are synthetic, not MIMIC excerpts.

## Citation

Küçükkara, M. Y. (2026). Context and Capacity Interact: Why Single-Factor Evaluations Misjudge
RAG-LLM ICD-10 Coding. *Manuscript under review.*

Software: archived on Zenodo at [10.5281/zenodo.21652988](https://doi.org/10.5281/zenodo.21652988)
(concept DOI, resolving to the latest version). The results in the paper correspond to release
v1.1.0: [10.5281/zenodo.21728018](https://doi.org/10.5281/zenodo.21728018).

## License

MIT (see `LICENSE`). MIMIC-IV itself is governed by the PhysioNet Credentialed Health Data Use
Agreement and is not covered by this license.
