# Figure captions (submission copy)

**Figure 1. The RAG-LLM ICD-10 coding pipeline under study, with evaluation-integrity safeguards.**
Candidate codes are retrieved from an ICD knowledge base (BM25, dense, or hybrid), note-local
evidence is retrieved for each candidate, all candidates are scored in one batched LLM call, and
an evidence constraint, contrastive verification, and a confidence threshold produce the final
code set. Dashed arrows mark the two safeguards described in Section 3.7: note-local evidence
retrieval, which removes cross-patient evidence provenance by construction, and a schema-robust
ranking key, which prevents a model that omits the confidence field from corrupting top-k
selection. Note that the scorer receives per-candidate evidence passages, not the discharge note
itself (Table M1).

**Figure 2. Component decomposition on the Top-50 label space (n = 17,151).**
Micro-F1 for the fourteen-arm ladder, coloured by system family. The dashed line marks the TF-IDF
baseline (E1). Classical and neural supervised arms (E1–E3) lead; retrieval-only, RAG, and
full-model arms cluster far below; generative LLM-only arms (E7, E8) are effectively inert because
they volunteer almost no codes. All thirteen non-E1 arms are significantly worse than E1
(note-level paired bootstrap, p < 0.001).

**Figure 3. Capacity ablation at the context-starved operating point: evidence judgement improves
with model scale, end-task accuracy does not.** (a) The discriminative lift — the factor by which a code the model marks "supported"
is more likely to be correct than one it marks "unsupported" — rises significantly from 1.02 (3B)
to 1.49 (7B); the dashed line at 1.0 denotes no discrimination. (b) Paired differences in micro-F1
(7B − 3B) for the unconstrained RAG arm (E11) and the full model (E14); both 95% confidence
intervals span zero (n.s.). Fixed code, identical 200-note set, note-level paired bootstrap. Both arms
run in the configuration of Table M1, where the scorer never sees the note; Section 4.5 shows that
scale does raise accuracy once the note is supplied, so this panel bounds what capacity buys *under
that starvation*, not in general.

**Figure 4. The global-corpus evidence design provides no patient-specific grounding.**
(a) Schematic: a code-description query is issued against an index holding one document per note
across all splits, so the top-ranked document is drawn from the corpus at large rather than from
the admission being coded. (b) Measured over 10,338 retrieved documents (1,000 notes, 5,169 code
queries) at the pipeline's own two-passage setting, 100% came from a different admission; the rate is
identical at k = 5 over 25,845 documents. With one document per note, chance alone predicts 0.99996,
so the rate confirms rather than discovers the behaviour; the finding is that no mechanism in the
design prefers the coded note's own text. This design was replaced by note-local retrieval before
the reported runs, every one of which records `evidence_note_local: true` (§4.3).

**Figure S1. Scalability across label-space size.**
Micro-F1 for four representative arms on Top-50, Top-100, and Top-200. The TF-IDF baseline holds
or improves with label-space size while every retrieval, RAG, and full-model arm degrades, so the
baseline's advantage over the full model widens monotonically from +0.317 to +0.397. Grey arrows
mark that gap at the endpoints.
