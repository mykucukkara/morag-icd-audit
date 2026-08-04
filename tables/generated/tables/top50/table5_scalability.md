### Table 5. Scalability across label spaces (micro-F1, single seed 42, full test split)

| System | Top-50 | Top-100 | Top-200 | Δ Top-50→Top-200 |
|---|---|---|---|---|
| E1 TF-IDF + LR | 0.4494 | 0.4686 | 0.4669 | +0.018 |
| E6 hybrid retrieval | 0.2027 | 0.1650 | 0.1234 | -0.079 |
| E11 hybrid RAG | 0.1865 | 0.1389 | 0.1096 | -0.077 |
| E14 full model | 0.1329 | 0.0981 | 0.0703 | -0.063 |
| **E1 lead over E14** | **+0.3165** | **+0.3705** | **+0.3966** | **widens** |
| Test notes (n) | 17,151 | 17,459 | 17,718 |  |
| Gold codes/note (mean) | 5.38 | 7.02 | 8.75 |  |
| Recall ceiling at a 15-code budget | 0.998 | 0.984 | 0.950 |  |
