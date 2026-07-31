# Checklist prevalence study — scoring matrix

Scored against the ten items of §5.4 under the rule fixed in
`manuscript/notes/checklist_scoring_protocol.md`. `Y` = the paper reports doing it; `N` = the
accessible text is specific enough to be confident it does not; `?` = undeterminable from the text
we could access; `–` = the item cannot apply because the system makes no claim of that kind (see
the amendment below).

## Amendment to the protocol, recorded during scoring (2026-07-30)

Two changes, both made before any prevalence figure was computed, both recorded here because the
protocol was written to stop exactly this kind of quiet adjustment:

1. **A `–` category was added.** Items 6 and 7 presuppose that the system retrieves evidence and
   displays it; item 8 presupposes that the model emits a structured object. A fine-tuned classifier
   with a sigmoid head has no schema to comply with and shows no evidence, so scoring it `N` would
   count a design choice as a reporting failure. Prevalence is therefore reported over the studies
   where the item *applies*, with the `–` count shown alongside. This shrinks denominators; it does
   not move any study from `N` to `Y`.
2. **Two abstract-stage includes were reversed on full text.** Recorded in the exclusions table
   below. Both reversals remove studies from the corpus on the pre-specified eligibility criteria,
   and one of them (Adrouji et al.) would have scored `Y` on item 9 — that is, the reversal works
   against this paper's thesis rather than for it.

## Checklist items (abbreviated)

| # | Item |
|---|---|
| 1 | note-blind floor reported and cleared |
| 2 | tuned classical/supervised baseline on the same split |
| 3 | oracle-over-shortlist and random-pruning decomposition |
| 4 | scorer input budget stated explicitly |
| 5 | context and capacity varied *jointly* |
| 6 | note-local evidence guaranteed / retrieval provenance reported |
| 7 | ≥1 grounding metric the model cannot self-report, reference string named |
| 8 | schema-compliance rate reported; no ranking-relevant default coercion |
| 9 | paired note-level significance test, resample count stated |
| 10 | machine-generated tables under provenance checks |

## Scores

### Studies read at full text (n = 8)

| Study | Evidence source | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Palacios et al. 2025, IEEE SDS (fine-tuned SLMs, MIMIC-IV Top-50) | full text (subscription) | N | Y | N | Y | N | – | – | – | N | N |
| Pathak et al. 2024, IEEE URTC (GPT-4o, CoT/RAG, HCUP) | full text (subscription) | N | N | N | N | N | Y | N | N | N | N |
| Krumscheid et al. 2025, SHTI/GMDS (Mistral-Small-24B RAG, German CARDIO:DE) | full text (open) | N | N | Y | Y | N | Y | N | N | N | N |
| Ong et al. 2023, J Med Artif Intell (ChatGPT, mockup retina encounters) | full text (open) | N | N | N | N | N | N | N | N | N | N |
| Hou et al. 2025, npj Health Systems (fine-tuned GPT-4o mini / Llama) | full text (open) | N | N | N | ? | Y | – | – | N | N | N |
| Jiang et al. 2026, Digital Health (5 LLMs × 5 knowledge-injection strategies) | full text (open) | N | N | N | Y | N | Y | N | N | Y | N |
| Schroeder et al. 2026, JAAOS (5 LLMs, hand-surgery notes) | full text (subscription) | N | N | N | N | N | – | N | – | N | N |
| Boyle et al. 2023, arXiv (off-the-shelf LLMs, CodiEsp-English) | full text (open) | N | Y | N | N | N | Y | N | N | N | N |

### Studies for which only the abstract (or abstract plus publisher article page) could be read (n = 12)

Every Methods-detail item is `?` here by construction — see the protocol's reasoning for why an
abstract cannot support an `N`. Item 2 is scored where the accessible text names the comparators.

| Study | Evidence source | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Klang et al. 2024, medRxiv (RAG-LLM ED ICD-10-CM vs human coders) | abstract | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| Li et al. 2024, arXiv (LLM multi-agents, MIMIC-III) | abstract | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| You et al. 2025, arXiv (MKE-Coder, Chinese EMRs) | abstract | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| Baksi et al. 2024, arXiv (MedCodER) | abstract | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| Dai et al. 2025, arXiv (section-aware fine-tuning, ICD-10-CM) | abstract | ? | N | ? | ? | ? | ? | ? | ? | ? | ? |
| Barreiros et al. 2025, CL4Health (explainable ICD coding via entity linking) | abstract | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| Yuan et al. 2025, EMNLP Industry (verification + lightweight adaptation) | abstract | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| Falis et al. 2024, JAMIA (can GPT-3.5 generate and code discharge summaries?) | abstract | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| Simmons et al. 2024, Appl Clin Inform (extracting ICD codes with LLMs) | abstract | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| Dayeh et al. 2026, SHTI (evidence-grounded LLM validation of MIMIC-IV labels) | abstract | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| Yoo & Kim 2025, Comput Biol Med (how to leverage LLMs for ICD coding) | abstract | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| Kaur et al. 2026, BMC MIDM (gold-standard-guided DPO, ICD-10-AM/ACHI) | abstract + article page | ? | N | ? | ? | ? | ? | ? | ? | ? | ? |

## Notes on individual scores

- **Ackermann et al. 2025 (IEEE SDS)** — item 2 scored `Y` generously: the paper places its own
  micro-F1 (0.70) beside a published supervised number on the same benchmark (MSMN, 0.74 on
  MIMIC-IV-ICD10-50) rather than re-running it. Item 5 scored `N` and is the sharpest external
  illustration of §4.5 in this corpus: context length and model capacity are confounded across the
  three models (Falcon-RW-1B at 2,048 tokens, Meditron-7B at 2,048, Phi3-Mini-3.5B at 3,072), and
  the paper concludes from that design that "longer context windows might be more valuable than
  larger, specialized models". With capacity and context varied one model at a time, neither
  conclusion is identified. Items 6–8 are `–`: the models carry a sigmoid multi-label head, so there
  is no retrieved evidence and no output schema.
- **Pathak et al. 2024 (IEEE URTC)** — item 6 scored `Y` because the retrieval corpus is stated
  explicitly (a 73,200-entry ICD-10 code list, cosine threshold 0.5, k = 20), which is the reporting
  half of the item; there is no cross-note retrieval to leak. The base condition scores 8.01% F1 and
  the best 47.12% on n = 50 patients with no floor and no classical comparator, so the reported
  39-point gain is not placed against any reference point other than the paper's own weakest prompt.

- **Krumscheid et al. 2025 (SHTI/GMDS)** — the only paper in the corpus that scores `Y` on item 3, and
  it earns it: a failure analysis attributes 61% of misclassifications to retrieval (the correct code
  was never in the candidate set), 25% to the LLM's final selection, and 14% to normalisation. That
  is the retrieval-ceiling-versus-selector separation item 3 asks for, arrived at independently and
  reaching the same conclusion as our §4.1c — the shortlist, not the judge, is where most of the loss
  is. What it does not have is an oracle-over-shortlist bound or a random-pruning null, so the
  selector's 25% share is not calibrated against what random selection would have produced. Item 4 is
  `Y`: the model "received the full clinical letter as context", with top-10 semantic plus top-10
  lexical candidates merged and deduplicated. Item 2 is `N`, and the comparison it does make is the
  pattern §5.3 warns about — its 76.3% top-1 *accuracy* on 228 German diagnoses is described as
  "promising" beside English MIMIC-III *micro-F1* "up to 76%", across a different language, dataset,
  metric and code cardinality.
- **Ong et al. 2023 (J Med Artif Intell)** — item 6 is `N` on unusually direct evidence. The
  encounters "were entered sequentially without starting a new chat", and the paper reports the
  consequence itself: errors "occurred sequentially", with laterality drifting late in the run, and
  notes this "may have been avoided by creating a new chat every time". Each note is therefore scored
  conditioned on every preceding note — a cross-note contamination of exactly the kind item 6 exists
  to rule out, here documented by the authors rather than hidden.
- **Hou et al. 2025 (npj Health Systems)** — the only `Y` on item 5, and the closest external
  corroboration of our §4.5 interaction. Table 1 crosses four model capacities (Llama-3.2-1B, -3B,
  Llama-3.1-8B, GPT-4o mini) against seven input conditions, and the capacity effect grows with input
  difficulty: on reordered diagnostic expressions the exact-match rate rises from 36.19% at 1B to
  78.23% at 8B, while all sizes sit within 3.85–10.90% on multiple concurrent conditions before
  enhanced tuning. The axis is input *form* rather than input *amount*, so it is not identical to our
  context factor, but it is a crossed design with an interaction, and it is the only one in the
  corpus. Item 9 is `N` because the arm-to-arm comparisons carry no paired test; the paper does report
  Wilson 95% confidence intervals and p-values on its headline rates, which makes it the only
  included study to attach interval estimates to its principal number.

- **Jiang et al. 2026 (Digital Health)** — the most carefully analysed study in the corpus, and the
  only one scoring `Y` on item 9: paired McNemar tests across three coding tasks with Bonferroni
  corrections applied separately to the prompt-strategy and model families, plus 95% confidence
  intervals on precision, recall, F1 and κ, and a blinded randomised review by five senior coders.
  Item 5 is nevertheless `N`, and the design is the clearest published instance of the problem §4.5
  is about: Phase I varies five knowledge-injection strategies on one model (GPT-4o), then Phase II
  fixes the winning strategy and varies five models. Capacity and context are each varied while the
  other is held constant, so no interaction between them is estimable — which matters here because
  the paper's headline conclusion is precisely about where the "performance boundaries" of knowledge
  injection lie. Item 8 is `N` and unmeasurable by construction: all runs went through "web-based
  user interfaces", so output format compliance was neither controlled nor recorded.
- **Schroeder et al. 2026 (JAAOS)** — item 9 is `N` on a specific ground rather than absence: the
  design is paired (the same 90 encounters go to every model and prompt) but "Chi-square analysis was
  done to assign statistical significance to these comparisons", which treats the arms as independent
  samples. The paper also states plainly that "No formal power analysis was done for this pilot
  study", which is the right disclosure to make.
- **Boyle et al. 2023 (arXiv)** — scores `Y` on item 2 and the comparison goes against the LLM:
  PLM-ICD reaches micro-F1 0.219 on CodiEsp-English against 0.138 for their GPT-4 tree search, with
  the LLM ahead only on macro-F1 (0.225 vs 0.216). Item 7 is `N` and is a missed opportunity the
  paper had in hand — CodiEsp carries span-level expert annotations that would have supported a
  grounding metric directly.

## Prevalence

Computed by `scripts/53_checklist_prevalence.py`, which parses this file so the reported figures and
the matrix cannot drift apart. Denominators are the studies where the item is determinable and
applicable; `?` and `–` counts are reported beside them rather than folded in.

## Excluded at full text after an abstract-stage include

| Record | Criterion | Reason |
|---|---|---|
| Prieto-Velasco et al. 2025, IEEE CBMS, "ICD code assignment from clinical text: impact of document composition" | 2 | The coder is a fine-tuned Spanish RoBERTa (BERTIN) with a classification head — solely a supervised encoder. The abstract's phrase "outputs of language models" carried the LLM signal that put it through abstract screening; the full text shows BERT-family fine-tuning over 26 arthropathy labels and a single principal diagnosis. Same category as the NorDeClin-BERT and MHLAT exclusions |
| Adrouji et al. 2026, SHTI, LLM ICD-10 coding vs. Thésorimed | 1 | The source text is regulatory (Summary of Product Characteristics drug labels), not clinical documentation of a patient. The pipeline is a genuine GPT-4o-mini RAG coder and would have scored `Y` on item 9 (two-level Cohen's κ with expected agreement reported), so this exclusion removes one of the corpus's more rigorous papers |
