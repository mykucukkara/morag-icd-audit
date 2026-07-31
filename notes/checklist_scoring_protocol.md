# Checklist prevalence study — protocol

Written before scoring, so the eligibility rule and the scoring rule cannot be adjusted to the
result. This answers the round-1 request (R2, item T3-3) for evidence behind the field-level claim
in §5.5: that the failure modes documented in this paper are not peculiar to our system.

## What this is and is not

It is a **bounded prevalence check**, not a systematic review. A full PRISMA sub-study is out of
scope for a single author without full-text access to most of the field, and doing one badly would
be worse than not doing one. The bound is stated in the reported result rather than hidden.

## Population

A study is eligible if it:

1. assigns ICD codes (any revision) to clinical free text, and
2. uses a large language model or retrieval-augmented generation as the coding mechanism — not
   solely a supervised encoder — and
3. makes an accuracy or explainability claim about its own system, and
4. is a full paper (not an abstract, editorial, poster or protocol).

Excluded: reviews and benchmarks that do not build a coder; studies where the ICD component cannot
be separated from another task; duplicates of the same system by the same group.

## Sampling frame

Three sources, pooled and deduplicated by DOI:

- the LLM/RAG ICD-coding studies already cited in this manuscript;
- the searches logged in `manuscript_litreview/search/SEARCH_LOG.md`, re-run with the same queries;
- targeted searches for the 2025–2026 lines identified during the round-3 fact-check.

This is a convenience frame, not an exhaustive one, and is reported as such.

## Scoring

Each study is scored against the ten items of the §5.4 checklist as:

| Code | Meaning |
|---|---|
| `Y` | the paper reports doing it |
| `N` | the paper reports enough for us to be confident it does not |
| `?` | cannot be determined from the text we could access |

**The `?` category is the point of this design.** Most checklist items concern Methods detail that
an abstract never carries. Scoring `N` from an abstract would manufacture a finding: absence of
mention is not absence of practice. So `N` is only assigned where the accessible text is specific
enough to support it, and every study records which text was used (`abstract`, `full text (open)`,
`full text (subscription)`).

Per item we report the prevalence among studies where the item could be determined, **and** the
number where it could not. A checklist item whose `?` count dominates is reported as undeterminable
in this frame rather than as evidence of non-reporting.

## Rater

Single rater (the author). No inter-rater reliability is available; this is stated as a limitation
rather than approximated by a second pass from the same person.

## Pre-specified expectation

Items 1, 3 and 5 (note-blind floor, oracle/null decomposition, joint context × capacity variation)
are contributions of this paper and are expected to be near-zero. That expectation is recorded here
so that finding them at zero is not presented as a discovery. The informative items are 2, 6, 7, 8
and 9 — a tuned classical baseline, evidence provenance, a grounding metric the model cannot
self-report, schema-compliance reporting, and note-level paired significance testing.

## Output

- `manuscript/notes/checklist_scoring_matrix.md` — one row per study, with the evidence source.
- A prevalence table in the supplement, and at most three sentences in §5.5.
