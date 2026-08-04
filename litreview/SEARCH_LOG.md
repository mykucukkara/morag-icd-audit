# Arama Günlüğü (PRISMA akış sayılarının TEK kaynağı)

Araç: academic-search MCP. Her satır bir çalıştırma. Sayılar buradan PRISMA diyagramına
(`figures/`) taşınır. Tekilleştirme DOI düzeyinde; kümülatif benzersiz havuz aşağıda.

> NOT (kalıcı): Semantic Scholar her çağrıda HTTP 429 (rate limit) verdi — kapsama boşluğu.
> Gönderim öncesi S1–S8 tekrar koşulup S2'nin dahil olduğu doğrulanmalı; Limitations'a işlendi.

## Çalıştırmalar

| # | Tarih | Sorgu | Kaynaklar | Param | Dönen | Elenen (min_alaka) | Konu-içi (kaba) |
|---|---|---|---|---|---|---|---|
| keşif-1 | 2026-07-23 | automated ICD-10 coding discharge summaries | OA,CR,S2*,ERIC,TRD | limit5, alaka0.6 | 5 | 5 | 5 (bağlam ağırlıklı) |
| S1 | 2026-07-23 | large language model ICD coding | OA,CR,S2*,ERIC,TRD | limit30, alaka0.6, yıl≥2020, sort=relevance | 30 | 6 | ~17 korpus adayı |
| S2 | 2026-07-23 | retrieval augmented generation ICD coding | OA,CR,S2*,ERIC,TRD | limit30, alaka0.6, yıl≥2020, sort=relevance | 30 | 6 | ~11 korpus adayı |
| S3 | 2026-07-23 | GPT ICD code assignment clinical notes | OA,CR,S2*,ERIC,TRD | limit30, alaka0.6, yıl≥2020, sort=relevance | 20 | 11 | ~6 yeni (çoğu klasik/nöral → bağlam) |
| S4 | 2026-07-23 | LLM medical coding discharge summaries | OA,CR,**S2 OK**,ERIC,TRD | aynı | 30 | 15 | ~12 yeni korpus adayı |
| S5 | 2026-07-23 | LLM ICD coding evaluation benchmark | OA,CR,S2*,ERIC,TRD | aynı | 30 | 9 | ~8 yeni (benchmark/agentic) |
| S6 | 2026-07-23 | explainable evidence ICD coding language model | OA,CR,**S2 OK**,ERIC,TRD | aynı | 30 | 22 | ~10 yeni (RQ3 ağırlıklı) |
| S7 | 2026-07-23 | ICD coding evidence extraction annotation | OA,CR,**S2 OK**,ERIC,TRD | aynı | 30 | 20 | ~7 yeni (RQ3/veri kalitesi) |
| S8 | 2026-07-23 | zero-shot few-shot ICD classification LLM | OA,CR,S2*,ERIC,TRD | aynı | 25 | 7 | **~0 — sorgu çöktü** (alan-dışı sınıflandırma gürültüsü: hacker/moda/termit/atık) |

*S2 = Semantic Scholar; S1–S3,S5,S8'de 429 (rate limit) döndü, ama S4/S6/S7'de ÇALIŞTI ve
yeni kayıtlar getirdi (ör. Falis 2024 JAMIA, MKE-Coder, Li multi-agent). Kapsama kısmen telafi
oldu; yine de tüm blokları S2 aktifken bir kez daha koşmak gerekli.

**S8 notu:** "zero-shot few-shot ICD classification LLM" sorgusu ICD-dışı sonuç döndürdü
(kaynaklar "ICD"yi in-context-demonstration kısaltmasıyla veya genel sınıflandırmayla
karıştırdı). Gönderim öncesi daha dar terimle tekrar: `"ICD coding" zero-shot LLM` / scope=baslik.

## ⚠️ KRİTİK BULGU — alanda zaten yayımlanmış sistematik derlemeler var

Aramalar iki mevcut derleme çıkardı; bu, "ilk derleme" novelty iddiasını GEÇERSİZ kılar ve
denetim (RQ4) açısını farklılaştırıcı olmaktan çıkıp ZORUNLU hale getirir:
- **Khalid, Ali, Fathima, Carole, Kim (2026).** "Recent Advances in AI for Automated ICD Coding:
  A Systematic Literature Review." *Journal of Medical Systems.* 10.1007/s10916-026-02429-7 — GÜNCEL, doğrudan rakip.
- **Kaur, Ginige, Obst (2023).** "AI-based ICD coding … A systematic literature review." *ESWA.*
  10.1016/j.eswa.2022.118997 — önceki dönem.
→ Karar: novelty'yi "LLM/RAG dönemi + değerlendirme-pratiği denetimi" olarak yeniden konumla.
  Bu iki derlemeyi Intro'da açıkça karşılaştır (ne kapsıyorlar, RQ4 denetimini yapıyorlar mı).
  Ayrıntı: `../notes/scope_decisions.md` Karar 2 güncellendi.

## Kümülatif benzersiz korpus adayları (S1+S2, DOI-tekilleştirilmiş) — TARAMA BEKLİYOR

Bunlar başlık/öz taramasına girecek "muhtemelen konu-içi" LLM/RAG-ICD çalışmaları.
Tam tarama kararları `../screening/screening_decisions.md`'ye işlenecek; burada yalnız havuz.

1. Ong 2023 — LLM (ChatGPT) retina ICD kodlama — 10.21037/jmai-23-106
2. Yoo & Kim 2025 — How to leverage LLMs for automatic ICD coding (CBM) — 10.1016/j.compbiomed.2025.109971
3. Li 2025 — Verification is All You Need: zero-shot clinical coding (JBHI) — 10.1109/jbhi.2025.3593028
4. Boukhers 2024 — LLM direct classification + enhanced text rep, ICD (BIBM) — 10.1109/bibm62325.2024.10822419
5. Barreiros 2025 — Explainable ICD Coding via Entity Linking (CL4Health) — 10.18653/v1/2025.cl4health-1.18  [RQ3]
6. Mustafa 2025 — LLMs vs human classifying clinical documents (IJMI) — 10.1016/j.ijmedinf.2025.105800
7. Hou 2025 — domain-specific fine-tuned LLMs, medical coding (npj Health Syst) — 10.1038/s44401-025-00018-3
8. Naliyatthaliyazchayil 2025 — reasoning capabilities of LLMs for medical coding, zero-shot (JMIR) — 10.2196/74142
9. Klang 2024 — RAG-LLM ED ICD-10-CM vs human coders (medRxiv) — 10.1101/2024.10.15.24315526  [RAG]
10. Kwan 2024 — LLMs good coders if given tools; Retrieve-Rank (arXiv) — 10.48550/arxiv.2407.12849  [RAG]
11. Singh 2025/2026 — MAX-EVAL-11 full-spectrum ICD-11 benchmark (BioNLP + medRxiv, DEDUP) — 10.18653/v1/2026.bionlp-1.23 / 10.1101/2025.10.30.25339130  [RQ2 benchmark]
12. Santos 2026 — LLMs ICD-10 obstetric Portuguese vs humans (preprint) — 10.21203/rs.3.rs-8712058/v1
13. Palacios 2025 — fine-tuned SLMs for ICD-10 (IEEE SDS) — 10.1109/sds66131.2025.00028
14. Simmons 2024 — extracting ICD-10-CM via LLMs (Appl Clin Inform) — 10.1055/a-2491-3872
15. Suvirat 2023 — leveraging LMs for inpatient diagnosis coding (Appl Sci) — 10.3390/app13169450
16. Lee & Lindsey 2024 — can LLMs abstract medical coded language? (arXiv) — 10.48550/arxiv.2403.10822
17. Alickovic 2026 — To RAG or Not to RAG: German tumor ICD (medRxiv) — 10.64898/2026.05.27.26353695  [RAG]
18. Puts 2024 — ICD-10 coding assistant GPT-4 + RoBERTa (JMIR Formative; preprint dup 10.2196/preprints.60095) — 10.2196/60095  [RAG]
19. Sarvari 2025 — LLMs guideline-based mgmt + automated coding (JMIR Biomed Eng) — 10.2196/66691
20. Abdaoui 2025 — BioClinicalBERT + RAG code mapping, anatomopathology (Bioengineering) — 10.3390/bioengineering13010030  [RAG, sınır]
21. Krumscheid 2025 — RAG for ICD-10 German clinical texts, case report (SHTI) — 10.3233/shti251397  [RAG]
22. Lehmann 2025 — RAG semantic annotation clinical trials ICD-10 (GMS) — 10.3205/25gmds061  [RAG, sınır: not-kodlama değil]
23. Adrouji 2026 — LLM ICD-10 drug contraindication concordance (SHTI) — 10.3233/shti260902  [sınır]
24. Hartnett 2026 — LLMs extract/translate + billing codes (arXiv) — 10.48550/arxiv.2603.22625  [sınır]
25. Li W. 2025 — Generation-Augmented Retrieval + guidelines, ICD diagnosis (JBHI) — 10.1109/jbhi.2025.3641931  [sınır: tanı vs kodlama]
26. Pathak 2024 — LLMs predict ICD-10 from records (IEEE URTC) — 10.1109/urtc65039.2024.10937521

## Yeni korpus adayları (S3–S8, DOI-tekilleştirilmiş, S1–S2 ile örtüşenler çıkarıldı) — TARAMA BEKLİYOR

Evidence/explainability ağırlıklı (RQ3 için değerli):
27. Beckh 2025 — "The Anatomy of Evidence: An Investigation Into Explainable ICD Coding" (ACL Findings) — 10.18653/v1/2025.findings-acl.864  [RQ3 çekirdek]
28. You 2025 — MKE-Coder: multi-axial knowledge + evidence verification, Chinese EMR — 10.48550/arxiv.2502.14916  [RQ3]
29. Ren 2025 — TraceCoder: traceable ICD via multi-source knowledge — 10.48550/arxiv.2510.15267  [RQ3]
30. Baksi 2024 — MedCodER: generative AI medical coding assistant (evidence) — 10.48550/arxiv.2409.15368  [RAG/RQ3]
31. Zhang 2026 — "From Documents to Spans: Evidence-Based ICD Coding with LLMs" — (DOI yok, S2 kaydı) [RQ3]
32. Li M. 2026 — "Evaluation and LLM-Guided Learning of ICD Coding Rationales" (EACL) — 10.18653/v1/2026.eacl-long.232  [RQ3/RQ4]
33. Wang Y. 2026 — "Interpretable ICD Code Classification with Faithful Sentence Extraction" (BioNLP) — 10.18653/v1/2026.bionlp-1.54  [RQ3]
34. Jiang 2024 — "Evidence Extraction for Automated Medical Coding: Preliminary Evaluation" — 10.1145/3711542.3711580  [RQ3]

LLM/RAG-ICD sistemleri (RQ1/RQ2):
35. Falis 2024 — "Can GPT-3.5 generate and code discharge summaries?" (JAMIA) — 10.1093/jamia/ocae132
36. Li R. 2024 — "Exploring LLM Multi-Agents for ICD Coding" (arXiv) — 10.48550/arxiv.2406.15363
37. Li R. 2025 — "Improving Rare and Common ICD Coding via a Multi-Agent LLM-Based Approach" (CIKM) — 10.1145/3746252.3760894
38. Yuan 2025 — "Toward Reliable Clinical Coding with LMs: Verification and Lightweight Adaptation" (EMNLP-industry) — 10.18653/v1/2025.emnlp-industry.12  [RQ4]
39. Simmons 2024b — "Benchmarking LLMs for Extraction of ICD Codes…" (medRxiv, Simmons 2024 yayınının ön-sürümü olabilir — TARAMADA DEDUP) — 10.1101/2024.04.29.24306573  [RQ2/RQ4]
40. Gan 2026 — "Enhancing LLM Medical Coding with Structured External Knowledge" (arXiv) — 10.48550/arxiv.2605.27377  [RAG]
41. Akkhawatthanakun 2026 — "Integrating Agentic AI to Automate ICD-10 Medical Coding" (Informatics) — 10.3390/informatics13030039
42. Bilioni 2025 — "Enhancing ML Models for Medical Coding: … Synthetic Clinical Notes" (IEEE CIBCB) — 10.1109/cibcb66090.2025.11177075
43. Bi 2024 — "Harmonising the Clinical Melody: Tuning LLMs for Hospital Course Summarisation in Clinical Coding" (arXiv) — 10.48550/arxiv.2409.14638
44. Vassileva 2025 — "Using LLMs for Multilingual Clinical Entity Linking to ICD-10" (RANLP) — 10.26615/978-954-452-098-4-151
45. Mustafa 2026 — "Can reasoning LLMs enhance clinical document classification?" (Health & Technology) — 10.1007/s12553-025-01041-y
46. Coutinho 2026 — "ICD coding of death certificates with generative language models" (PLOS Digital Health) — 10.1371/journal.pdig.0001245
47. Vo 2025 — "Synthetic Clinical Notes for Rare ICD Codes … Long-Tail Coding" (arXiv) — 10.48550/arxiv.2511.14112
48. Lenz 2025 — "Can open source LLMs be used for tumor documentation in Germany?" (BioData Mining) — 10.1186/s13040-025-00463-8
49. Jaganathan 2025 — "Metamorphic Testing for Robustness and Fairness Eval of LLM-based ICD Coding" (Smart Health) — 10.1016/j.smhl.2025.100564  [RQ4]
50. Nawab 2024 — "Fine-tuning … GPT for automatic assignment of ICD codes" (J Med AI; RS ön-sürüm dup) — 10.21037/jmai-24-60
51. Chen Z. 2024 — "Zero-Shot ATC Coding with LLMs" (arXiv) — 10.48550/arxiv.2412.07743  [sınır: ATC, ICD değil → muhtemelen HARIC]
52. Palacios/Carberry aileleri — Carberry hierarchical/explainable LLM ICD (10.62791/1994) [RQ3]

Veri-kalitesi / benchmark / MIMIC-etiket-gürültüsü (RQ2/RQ4 + makale-1 ile tema akrabası):
53. Dayeh 2026 — "Evidence-Grounded LLM Validation of MIMIC-IV ICD Labels" (SHTI) — 10.3233/shti260328  [RQ3/RQ4; MIMIC etiket gürültüsü]
54. Khadka 2025 — "Data Quality in Clinical Coding: A Critical Analysis" (medRxiv) — 10.1101/2025.08.24.25334321  [RQ4/bağlam]
55. Yang 2026 — MedicalBench / MIMIC-IV-Ext (concept extraction benchmark) — 10.13026/j98m-g356 (+ arXiv/medRxiv dup)  [sınır: kavram çıkarımı]

NOT: 51, 55 sınır — tarama uygunluk ölçütleriyle (ICD KOD ATAMASI mı?) elenebilir.
Klasik/nöral açıklanabilir kodlama (HiLAT, TransICD, MHLAT, ICDXML, LAJA, PLM-ICD, Pascual 2021,
Teng 2024 few-shot-evidence) → çoğu BAĞLAM (üretken LLM değil); "açıklanabilir kodlama"
kökeni olarak Related Work'te anılır, korpusa girmez.

## Snowballing madeni: Gershon 2025 bibliyografyası (37 kaynak, hepsi ICD-LLM) — OKUNDU

Gershon'un dahil ettiği 35 çalışma bizim korpusumuzla büyük ölçüde çakışacak (meşru — yeni mercek).
Tarama sürecine girecek, kanıt/RAG-ilgili YENİ (havuzumuzda olmayan) DOI'ler:
56. Niu, Wu, Li, Li 2023 — "Retrieve and rerank for automated ICD coding via Contrastive Learning" (JBI) — 10.1016/j.jbi.2023.104396  [RAG/retrieve-rerank — çekirdek]
57. Blanco 2022 — PlaBERT, per-label attention, çok-dilli (JBI) — 10.1016/j.jbi.2022.104050  [explainability/RQ1]
58. Falissard 2022 — Neural Translation ICD-10 entities (JMIR MI) — 10.2196/26353  [seq2seq]
59. Coutinho & Martins 2022 — ICD-10 death certs Portuguese (JBI) — 10.1016/j.jbi.2022.104232  [Coutinho 2026 ile aynı ekip]
60. Lamproudis 2023 — Large open clinical corpus ICD-10 (AMIA) — (DOI yok, AMIA proc)  [İsveççe]
61. Wang Y. 2024 — "Validation of GPT-4 for clinical event classification vs ICD + human" (JGH) — 10.1111/jgh.16561  [insan-karşılaştırmalı — RQ2]
62. Abdelgadir 2024 — ChatGPT nephrology ICD-10 (Front AI) — 10.3389/frai.2024.1457586  [prompt-tabanlı]
63. Zambetta 2024 — hybrid DNN+rule French death certs (IJMI) — 10.1016/j.ijmedinf.2024.105462
64. Lu & Xue 2023 — transformer+GCN ICD (Knowl-Based Syst) — 2023;282:111113  [bağlam/borderline]

Gershon'un kalan kaynakları çoğunlukla BERT-ailesi encoder (PLM-ICD hattı) → bizim üretken-LLM/RAG+
kanıt-iddiası ölçütümüzde BAĞLAM. HiLAT/PLM-ICD/ICDXML vb. Related Work'te "explainable coding kökeni".
NOT: Gershon'da HiLAT (Liu 2022, 10.1016/j.jbi.2022.104161) explainable-attention örneği — RQ1 bağlamı.

## Bağlam kaynağı olarak ayrılanlar (korpus DIŞI — Intro/Related için)

- PLM-ICD (Huang 2022) 10.18653/v1/2022.clinicalnlp-1.2 — PLM hattı, üretken LLM değil → bağlam/baseline
- Chen 2021 (JMIR MI) 10.2196/23230 — denetimli derin ağ ICD-10 → bağlam
- Shuai 2022 (BMC MIDM) 10.1186/s12911-022-01753-5 — özellik çıkarımı karşılaştırması → bağlam
- Kane 2023 (BMC Bioinformatics) 10.1186/s12859-023-05597-2 — ICD-10-CM embedding veri kümesi → kaynak/altyapı
- Kaur 2023 (ESWA) 10.1016/j.eswa.2022.118997 — önceki dönem sistematik derleme → konumlandırma
- **Khalid 2026 (J Med Syst) 10.1007/s10916-026-02429-7 — GÜNCEL AI-ICD sistematik derlemesi → doğrudan konumlandırma/rakip**
- Klasik açıklanabilir kodlama temelleri: Liu 2022 HiLAT (JBI) 10.1016/j.jbi.2022.104161, Biswas 2021 TransICD,
  Duan 2023 MHLAT, Pascual 2021 BERT-ICD limitations → "explainable coding" kökeni, Related Work

## Kapsam-dışı (elenecekler — kodlama görevi değil)

SDoH çıkarımı, kas-iskelet/nöbet/dementia/T2DM/ASCVD fenotipleme, irAE tespiti, RadOnc-GPT,
mental sağlık tanısı, interoperabilite, EDI-837, genel RAG makaleleri (kütüphane/yangın/RAG
ders kitabı bölümleri), lncRNA. Gerekçe: ICD kod ATAMASI değil.
