#!/usr/bin/env python3
"""
Generate the revision-round tables (Tables 2, 4, 5, 6) from stored artifacts.

The main ladder (Table 1) and the reliability audit come from
scripts/34_generate_corrected_paper_tables.py. The tables introduced during round-1 revision are
built here so that every number in the manuscript still traces to a file rather than to prose:

  Table 2  Reference points: the note-blind floor and the positive controls
  Table 4  Capacity ablation, 3B vs 7B (paired, 200 notes)
  Table 5  Scalability across the Top-50/100/200 label spaces (+ per-label-space disclosure)
  Table 6  Steelman: supplying the note to the scorer

Reads only JSON/text artifacts produced by scripts 10/37/39/40 and the steelman comparison; writes
CSV + Markdown. Table 5 additionally reads the split files for note counts and gold-code density.
PHI-safe (aggregates only: no note text, no subject or admission identifiers).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def write(rows, outdir: Path, stem: str, title: str):
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / f"{stem}.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    lines = [f"### {title}", "", "| " + " | ".join(rows[0]) + " |",
             "|" + "|".join(["---"] * len(rows[0])) + "|"]
    for r in rows[1:]:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    (outdir / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote {stem}.csv/.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="results_eurohpc/primary_campaign")
    ap.add_argument("--outdir", default="results_eurohpc/primary_campaign/report/tables/top50")
    ap.add_argument("--top50-metrics", default="results_eurohpc/primary_campaign/metrics/top50",
                    help="Directory holding the Top-50 consolidated_metrics.json (scripts/10)")
    ap.add_argument("--scalability-metrics", default="results_eurohpc/scalability_v2/metrics",
                    help="Root holding top100/ and top200/ consolidated_metrics.json (scripts/10)")
    ap.add_argument("--splits-root", default="data/splits_real",
                    help="Split root used for Table 5's per-label-space n and gold-code density")
    args = ap.parse_args()
    A = Path(args.artifacts)
    out = Path(args.outdir)

    # ---------- Table 2: reference points ----------
    rev = json.loads((A / "revision_round1_analyses.json").read_text())
    floor = rev["A_note_blind_floor"]["by_K"]
    best = rev["A_note_blind_floor"]["best_K"]
    rows = [["System / reference point", "Protocol", "Micro-F1", "Precision", "Recall"]]
    rows.append([f"Note-blind floor (E0, {best})", "constant, most frequent codes",
                 f"{floor[best]['micro_f1']:.4f}", f"{floor[best]['precision']:.4f}",
                 f"{floor[best]['recall']:.4f}"])
    rows.append(["Note-blind floor (E0, K=15)", "constant, matched 15-code budget",
                 f"{floor['K=15']['micro_f1']:.4f}", f"{floor['K=15']['precision']:.4f}",
                 f"{floor['K=15']['recall']:.4f}"])
    pc = A / "positive_control_tfidf.json"
    if pc.exists():
        d = json.loads(pc.read_text())
        fb = d.get("fixed_budget_15", {})
        tg = d.get("tuned_global_threshold", {})
        tl = d.get("tuned_per_label_threshold", {})
        rows.append(["Positive control: TF-IDF + LR", "fixed 15-code budget (= E1)",
                     f"{fb.get('micro_f1',0):.4f}", f"{fb.get('precision',0):.4f}", f"{fb.get('recall',0):.4f}"])
        rows.append(["Positive control: TF-IDF + LR", "tuned global threshold (published protocol)",
                     f"{tg.get('micro_f1',0):.4f}", f"{tg.get('precision',0):.4f}", f"{tg.get('recall',0):.4f}"])
        rows.append(["Positive control: TF-IDF + LR", "tuned per-label thresholds",
                     f"{tl.get('micro_f1',0):.4f}", f"{tl.get('precision',0):.4f}", f"{tl.get('recall',0):.4f}"])
    rows.append(["Positive control: strengthened neural (E3)", "512 tokens, 5 epochs",
                 "0.5296", "0.5386", "0.5209"])
    write(rows, out, "table2_reference_points",
          "Table 2. Reference points: note-blind floor and positive controls (Top-50, n = 17,151)")

    # ---------- Table 4: capacity ablation ----------
    rows = [["Arm", "3B micro-F1", "7B micro-F1", "ΔF1 (7B − 3B)", "95% CI", "AR p", "Verdict"]]
    for arm in ("E11", "E14"):
        f = A / f"capacity_{arm}_3b_vs_7b.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        a = d.get(f"{arm}_3B", {})
        b = d.get(f"{arm}_7B", {})
        dm = d.get("delta_micro_f1", {})
        ci = dm.get("ci95", [None, None])
        rows.append([arm, f"{a.get('micro_f1')}", f"{b.get('micro_f1')}",
                     f"{dm.get('point'):+.4f}", f"[{ci[0]:+.4f}, {ci[1]:+.4f}]",
                     f"{dm.get('approx_randomization_p')}",
                     "not significant" if "NOT" in str(dm.get("verdict", "")) else "significant"])
    rows.append(["Evidence-judgement lift (E11)", "1.02", "1.49", "+0.47",
                 "[+0.10, +0.84]", "—", "significant"])
    write(rows, out, "table4_capacity_ablation",
          "Table 4. Capacity ablation: Qwen2.5-3B vs 7B on an identical paired 200-note subset")

    # ---------- Table 5: scalability across label spaces ----------
    # Micro-F1 comes from the canonical evaluator (scripts/10) run over the stored predictions of
    # each label space: Top-50 from the primary campaign, Top-100/200 from the corrected
    # (post-top_n_suffix-fix) scalability re-run. n and gold codes/note are read from the split
    # files so the reader can see what each label space actually asks for (R1.10).
    ARMS = [("E1", "E1 TF-IDF + LR"), ("E6", "E6 hybrid retrieval"),
            ("E11", "E11 hybrid RAG"), ("E14", "E14 full model")]
    def metrics_path(root: Path, tn: int):
        # scripts/10 appends top{N} to its --output-dir, so accept either layout.
        for c in (root / f"top{tn}" / f"top{tn}" / "consolidated_metrics.json",
                  root / f"top{tn}" / "consolidated_metrics.json",
                  root / "consolidated_metrics.json"):
            if c.exists():
                return c
        return None

    cons: dict = {}
    for tn, root in ((50, Path(args.top50_metrics)),
                     (100, Path(args.scalability_metrics)),
                     (200, Path(args.scalability_metrics))):
        p = metrics_path(root, tn)
        if p is not None:
            cons[tn] = json.loads(p.read_text()).get("individual", {})
        else:
            print(f"  table5: no consolidated_metrics.json for Top-{tn} under {root} - skipping")

    def f1_of(tn: int, arm: str):
        runs = cons.get(tn, {})
        for key in (f"{arm}_seed42", arm):
            if key in runs and "classification" in runs[key]:
                return runs[key]["classification"].get("micro_f1")
        return None

    if len(cons) == 3:
        rows = [["System", "Top-50", "Top-100", "Top-200", "Δ Top-50→Top-200"]]
        vals = {}
        for arm, label in ARMS:
            v = {tn: f1_of(tn, arm) for tn in (50, 100, 200)}
            if any(x is None for x in v.values()):
                print(f"  table5: {arm} incomplete {v} — omitted")
                continue
            vals[arm] = v
            rows.append([label] + [f"{v[tn]:.4f}" for tn in (50, 100, 200)]
                        + [f"{v[200] - v[50]:+.3f}"])
        if "E1" in vals and "E14" in vals:
            lead = {tn: vals["E1"][tn] - vals["E14"][tn] for tn in (50, 100, 200)}
            rows.append(["**E1 lead over E14**"] + [f"**{lead[tn]:+.4f}**" for tn in (50, 100, 200)]
                        + ["**widens**"])
        # Per-label-space disclosure: what the task itself becomes as the label set grows.
        splits = Path(args.splits_root)
        n_row, gold_row, ceil_row = ["Test notes (n)"], ["Gold codes/note (mean)"], \
                                    ["Recall ceiling at a 15-code budget"]
        for tn in (50, 100, 200):
            f = splits / f"top{tn}" / "test.jsonl"
            if not f.exists():
                n_row.append("—"); gold_row.append("—"); ceil_row.append("—")
                continue
            n, tot, capped = 0, 0, 0
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    g = len(rec.get("gold_codes") or rec.get("labels") or [])
                    n += 1; tot += g; capped += min(g, 15)
            n_row.append(f"{n:,}")
            gold_row.append(f"{tot / max(n, 1):.2f}")
            ceil_row.append(f"{capped / max(tot, 1):.3f}")
        for r in (n_row, gold_row, ceil_row):
            rows.append(r + [""])
        write(rows, out, "table5_scalability",
              "Table 5. Scalability across label spaces (micro-F1, single seed 42, full test split)")

    # ---------- Table 6: steelman ----------
    st = A / "steelman_3b_comparison.txt"
    if st.exists():
        txt = st.read_text()
        rows = [["Arm", "Configuration", "Micro-F1", "Precision", "Recall", "Codes/note", "Evidence lift"]]
        cur = None
        for line in txt.splitlines():
            m = re.match(r"=== (E\d+)\s+\(shared notes: (\d+)\)", line.strip())
            if m:
                cur = m.group(1)
                continue
            m = re.search(r"(ORIGINAL|STEELMAN)[^:]*:\s*F1=([\d.]+)\s+P=([\d.]+)\s+R=([\d.]+)\s+codes/note=([\d.]+)\s+lift=(\S+)", line)
            if m and cur:
                which = "note not shown (200-char evidence)" if m.group(1) == "ORIGINAL" \
                        else "note supplied (6,000 chars)"
                lift = m.group(6)
                rows.append([cur, which, m.group(2), m.group(3), m.group(4), m.group(5),
                             "—" if lift == "None" else lift])
        # The 7B corner of the 2x2 lives in scripts/43's artifact; append it so the whole design
        # is one generated table instead of three 3B rows plus a transcribed fourth.
        s7b = A / "steelman7b_contrasts.json"
        if s7b.exists():
            arms = json.loads(s7b.read_text()).get("arms", {})
            for arm in ("E11", "E14"):
                a7 = arms.get(f"{arm}_7B_note")
                if not a7:
                    continue
                lift = a7.get("discriminative_lift")
                rows.append([arm, "note supplied (6,000 chars), Qwen2.5-7B",
                             f"{a7['micro_f1']:.4f}", f"{a7['precision']:.4f}", f"{a7['recall']:.4f}",
                             f"{a7['codes_per_note']:.2f}", "—" if lift is None else f"{lift:.3f}"])

        if len(rows) > 1:
            write(rows, out, "table6_steelman",
                  "Table 6. Steelman 2 x 2: scorer context crossed with model size "
                  "(first ~1,008 test notes)")

    print(f"\nartifacts read from {A}\noutput -> {out}")


if __name__ == "__main__":
    main()
