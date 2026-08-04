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
                    help="Split root for Top-50 (the reference partition)")
    #  Top-100/200 were rebuilt on the Top-50 subject partition (scripts/51), so their n and
    #  gold-code density come from a different tree than Top-50's. One flag would silently pair
    #  shared-partition scores with the superseded splits' counts.
    ap.add_argument("--splits-root-scalability", default=None,
                    help="Split root for Top-100/200; defaults to --splits-root")
    args = ap.parse_args()
    A = Path(args.artifacts)
    out = Path(args.outdir)
    splits_scal = Path(args.splits_root_scalability or args.splits_root)

    # ---------- Table 2: reference points ----------
    rows = [["System / reference point", "Protocol", "Micro-F1", "Precision", "Recall"]]

    #  The floor must be the validation-selected one (§4.1b). The first version picked the best K on
    #  test itself, which tunes the reference point on the data it is a reference for and, as it
    #  happens, produced a *lower* floor (K=10, 0.3040) that the systems had an easier time clearing.
    #  This table kept quoting that superseded value after the prose had moved on.
    r2 = A / "reviewer_round2_analyses.json"
    if r2.exists():
        fl = json.loads(r2.read_text())["A_floor_selected_on_validation"]
        k, t, m = fl["selected_K"], fl["test_once"], fl["test_at_matched_budget_K15"]
        rows.append([f"Note-blind floor (E0, K={k})", "constant; K selected on validation, scored once on test",
                     f"{t['micro_f1']:.4f}", f"{t['precision']:.4f}", f"{t['recall']:.4f}"])
        rows.append(["Note-blind floor (E0, K=15)", "constant, matched 15-code budget",
                     f"{m['micro_f1']:.4f}", f"{m['precision']:.4f}", f"{m['recall']:.4f}"])
    else:
        raise SystemExit("reviewer_round2_analyses.json missing — Table 2 would report the "
                         "superseded test-selected floor; refusing to build")
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
    #  The label-attention control is the architecture family the ICD-coding literature reports its
    #  best numbers with, and it was the strongest supervised reference the study has — yet it lived
    #  only in a Results paragraph while the weaker E3 sat in this table. Both protocols are shown
    #  because they answer different questions, and both are read from the artifact.
    lac = A / "label_attention_control.json"
    if lac.exists():
        d = json.loads(lac.read_text())
        fb, tt = d.get("fixed_budget", {}), d.get("tuned_threshold", {})
        enc = str(d.get("encoder", "?")).replace("_safetensors", "")
        rows.append([f"Positive control: label-attention (E3b, {enc})",
                     "fixed 15-code budget (= E1)",
                     f"{fb.get('micro_f1',0):.4f}", f"{fb.get('precision',0):.4f}", f"{fb.get('recall',0):.4f}"])
        rows.append(["Positive control: label-attention (E3b)",
                     f"threshold {tt.get('threshold','?')} selected on validation",
                     f"{tt.get('micro_f1',0):.4f}", f"{tt.get('precision',0):.4f}", f"{tt.get('recall',0):.4f}"])
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
        n_row, gold_row, ceil_row = ["Test notes (n)"], ["Gold codes/note (mean)"], \
                                    ["Recall ceiling at a 15-code budget"]
        #  These three rows are also written to an artifact. They are the only numbers in the paper
        #  that can be derived *only* from the split files, which are clinical text and can never be
        #  published — so on any machine without them the checks that verify §4.4's cohort figures
        #  simply do not run, and a check that does not run looks exactly like a check that passed.
        #  Emitting the aggregates (three scalars per label space, no text, no identifiers) lets the
        #  guard verify them in the public repository's own layout.
        disclosure: dict = {"note": ("Aggregates of the test splits behind Table 5's disclosure "
                                     "rows. Top-50 from the reference partition; Top-100/200 from "
                                     "the shared-partition rebuild (scripts/51)."),
                            "splits_root_top50": str(args.splits_root),
                            "splits_root_scalability": str(splits_scal),
                            "by_label_space": {}}
        for tn in (50, 100, 200):
            # Top-50 is the reference partition; Top-100/200 were rebuilt on it (scripts/51),
            # so their counts must come from the rebuilt tree, not the superseded splits.
            f = (Path(args.splits_root) if tn == 50 else splits_scal) / f"top{tn}" / "test.jsonl"
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
            #  Stored at six decimals, not four. The Top-100 ceiling is 0.98449…, which prints as
            #  0.984 in the table; rounded to four first it becomes 0.9845, and rounding *that* to
            #  the three decimals the prose uses gives 0.985. An artifact that is itself pre-rounded
            #  turns every downstream check into a double rounding.
            disclosure["by_label_space"][f"top{tn}"] = {
                "test_notes": n,
                "gold_codes_per_note": round(tot / max(n, 1), 6),
                "recall_ceiling_at_15": round(capped / max(tot, 1), 6),
                "source": str(f),
            }
        for r in (n_row, gold_row, ceil_row):
            rows.append(r + [""])
        if disclosure["by_label_space"]:
            (A / "scalability_disclosure.json").write_text(
                json.dumps(disclosure, indent=2), encoding="utf-8")
            print(f"  wrote {A / 'scalability_disclosure.json'}")
        write(rows, out, "table5_scalability",
              "Table 5. Scalability across label spaces (micro-F1, single seed 42, full test split)")

    # ---------- Table 6: steelman ----------
    st = A / "steelman_3b_comparison.txt"
    if st.exists():
        # Table 6 is built from `capacity_curve.json` (scripts/54), which holds all six cells of
        # the 3 x 2 design scored on one note set. The previous version scraped a text log for the
        # 3B rows and appended two 7B rows from a second artifact, so it silently stayed a 2 x 2
        # after the third capacity point was added — the table a reviewer read did not match the
        # design the text described.
        cc = A / "capacity_curve.json"
        if not cc.exists():
            print("  table6: capacity_curve.json missing — refusing to emit a partial design")
            rows = []
        else:
            curve = json.loads(cc.read_text())["arms"]
            rows = [["Arm", "Scorer context", "Model", "Micro-F1", "Precision", "Recall",
                     "Codes/note"]]
            ctx_label = {"nonote": "note withheld (200-char evidence)",
                         "note": "note truncated to 6,000 chars"}
            for arm in ("E11", "E14"):
                cells = curve.get(arm, {}).get("cells", {})
                for ctx in ("nonote", "note"):
                    for cap in ("3B", "7B", "14B"):
                        c = cells.get(f"{cap}_{ctx}")
                        if not c:
                            continue
                        rows.append([arm, ctx_label[ctx], f"Qwen2.5-{cap}",
                                     f"{c['micro_f1']:.4f}", f"{c['precision']:.4f}",
                                     f"{c['recall']:.4f}", f"{c['codes_per_note']:.2f}"])
        if len(rows) > 1:
            write(rows, out, "table6_steelman",
                  "Table 6. Scorer context crossed with model capacity (3 x 2), all cells on the same "
                  "1,008 notes (shards 0-3 of the 68-way split)")

    print(f"\nartifacts read from {A}\noutput -> {out}")


if __name__ == "__main__":
    main()
