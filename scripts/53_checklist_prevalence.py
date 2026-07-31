#!/usr/bin/env python3
"""
Script 53: prevalence of the §5.4 checklist items across the corpus (reviewer item T3-3).

Parses `manuscript/notes/checklist_scoring_matrix.md` rather than taking numbers from prose, so the
figures quoted in §5.5 and the supplement cannot drift away from the matrix a reader can inspect.
This is item 10 of the checklist applied to the checklist study itself.

Two denominators are reported per item, because they answer different questions:
  * `determinable` — studies where the item could be scored at all (Y + N). This is the prevalence
    figure, and for most items it is the eight full-text studies.
  * `all` — every eligible study, with the `?` and `–` counts shown. A reader who thinks `?` should
    count against the field can compute that; a reader who thinks it should not, can too.

The rule-of-three upper bound is reported for items observed at zero: with 0 successes in n trials,
the one-sided 95% upper bound on the true rate is approximately 3/n. It is the honest way to say
"we saw none" without implying "there are none".

Usage:
    python scripts/53_checklist_prevalence.py \
        --matrix manuscript/notes/checklist_scoring_matrix.md \
        --out results_eurohpc/primary_campaign/checklist_prevalence.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ITEM_LABELS = {
    1: "note-blind floor reported and cleared",
    2: "tuned classical/supervised baseline on the same split",
    3: "retrieval-ceiling vs selector decomposition",
    4: "scorer input budget stated",
    5: "context and capacity varied jointly",
    6: "note-local evidence / retrieval provenance reported",
    7: "grounding metric the model cannot self-report",
    8: "schema-compliance rate reported",
    9: "paired note-level significance test",
    10: "machine-generated tables under provenance checks",
}
VALID = {"Y", "N", "?", "–", "-"}


def parse(md: str) -> list[dict]:
    """Rows of the two score tables: any table row whose last ten cells are all score codes."""
    rows = []
    for line in md.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 12:
            continue
        scores = cells[2:]
        if not all(s in VALID for s in scores):
            continue
        rows.append({"study": cells[0], "source": cells[1],
                     "scores": ["–" if s == "-" else s for s in scores]})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="manuscript/notes/checklist_scoring_matrix.md")
    ap.add_argument("--out", default="results_eurohpc/primary_campaign/checklist_prevalence.json")
    args = ap.parse_args()

    rows = parse(Path(args.matrix).read_text(encoding="utf-8"))
    if not rows:
        raise SystemExit(f"no score rows parsed from {args.matrix} — has the table format changed?")

    fulltext = [r for r in rows if r["source"].startswith("full text")]
    print(f"  {len(rows)} eligible studies, {len(fulltext)} read at full text")

    items = {}
    for i in range(1, 11):
        col = [r["scores"][i - 1] for r in rows]
        y, n, q, na = (col.count("Y"), col.count("N"), col.count("?"), col.count("–"))
        det = y + n
        rec = {"item": i, "label": ITEM_LABELS[i], "Y": y, "N": n, "undeterminable": q,
               "not_applicable": na, "determinable": det,
               "prevalence_among_determinable": round(y / det, 3) if det else None,
               "corpus_size": len(rows)}
        if y == 0 and det:
            # 0/n: the one-sided 95% upper bound is ~3/n. Reported so "none observed" is not read
            # as "none exists" — the same courtesy this paper asks of the literature.
            rec["rule_of_three_upper_bound"] = round(3.0 / det, 3)
        items[str(i)] = rec
        bound = f"  (95% upper bound {rec['rule_of_three_upper_bound']})" if y == 0 and det else ""
        print(f"    item {i:>2}: {y}/{det} = "
              f"{rec['prevalence_among_determinable'] if det else 'n/a'}"
              f"   [? {q}, – {na}]{bound}")

    report = {
        "corpus_size": len(rows),
        "read_at_full_text": len(fulltext),
        "read_at_abstract_only": len(rows) - len(fulltext),
        "single_rater": True,
        "frame": "convenience frame; not a systematic review (see checklist_scoring_protocol.md)",
        "items": items,
        "items_with_zero_observed": sorted(int(k) for k, v in items.items() if v["Y"] == 0),
        "studies": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
