### Table 2. Reference points: note-blind floor and positive controls (Top-50, n = 17,151)

| System / reference point | Protocol | Micro-F1 | Precision | Recall |
|---|---|---|---|---|
| Note-blind floor (E0, K=10) | constant, most frequent codes | 0.3040 | 0.2338 | 0.4345 |
| Note-blind floor (E0, K=15) | constant, matched 15-code budget | 0.2850 | 0.1936 | 0.5398 |
| Positive control: TF-IDF + LR | fixed 15-code budget (= E1) | 0.4491 | 0.3051 | 0.8505 |
| Positive control: TF-IDF + LR | tuned global threshold (published protocol) | 0.6053 | 0.5852 | 0.6267 |
| Positive control: TF-IDF + LR | tuned per-label thresholds | 0.5982 | 0.5349 | 0.6785 |
| Positive control: strengthened neural (E3) | 512 tokens, 5 epochs | 0.5296 | 0.5386 | 0.5209 |
