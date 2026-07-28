#!/usr/bin/env python3
"""
Script 45: assemble the manuscript into submission artifacts.

Reads the section files, the generated tables and the figure captions, and writes:

  build/MORAG-ICD_manuscript.md   one Markdown file, author-year citations as written
  build/main.tex                  LaTeX with numbered citations (\\cite), tables and figures
  build/references.bib            copy of the bibliography, next to main.tex

The Markdown stays the editing format; the LaTeX is generated, never hand-edited, so the two
cannot drift. Citations are the delicate part: the prose is written author-year and the target
journals want numbered references, so every "(Author et al., 2024)" and "Author et al. (2024)" is
resolved against the bib keys. An unresolved citation is a hard error rather than a silent
passthrough — a dangling \\cite prints as "[?]" in the PDF and is exactly the kind of defect this
paper is about.

Usage:
    python scripts/45_build_manuscript.py --manuscript manuscript --out manuscript/build
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from pathlib import Path

# Section files in reading order, with the LaTeX sectioning command each needs.
SECTIONS = [
    ("02_introduction.md", "Introduction"),
    ("03_related_work.md", "Related Work"),
    ("04_methods.md", "Methods"),
    ("05_results.md", "Results"),
    ("06_discussion.md", "Discussion"),
    ("07_limitations_future_work.md", "Limitations and Future Work"),
    ("08_conclusion.md", "Conclusion"),
]

UNICODE = {
    "—": "---", "–": "--", "−": "$-$", "→": "$\\rightarrow$", "×": "$\\times$",
    "≈": "$\\approx$", "≤": "$\\leq$", "≥": "$\\geq$", "±": "$\\pm$",
    "Δ": "$\\Delta$", "α": "$\\alpha$", "μ": "$\\mu$", "§": "\\S~", "·": "$\\cdot$",
    "ü": '\\"u', "Ü": '\\"U', "ö": '\\"o', "Ö": '\\"O', "ç": "\\c{c}", "Ç": "\\c{C}",
    "ğ": "\\u{g}", "Ğ": "\\u{G}", "ş": "\\c{s}", "Ş": "\\c{S}", "ı": "\\i{}",
    "İ": "\\.{I}", "’": "'", "‘": "`", "“": "``", "”": "''",
}


def strip_working_notes(md: str) -> str:
    """Remove HTML comments.

    Markdown renderers hide them, so they read as private notes-to-self while drafting — and one
    ("NOTE (working file, remove before submission)") duly reached the assembled Markdown and the
    Word file, where nothing hides them.
    """
    return re.sub(r"<!--.*?-->", "", md, flags=re.S)


def load_citation_map(bib_path: Path) -> dict[tuple[str, str], str]:
    """(first-author surname lowercased, year) -> bib key."""
    text = bib_path.read_text(encoding="utf-8")
    out: dict[tuple[str, str], str] = {}
    for entry in re.split(r"\n(?=@)", text):
        m = re.match(r"@\w+\{([^,]+),", entry.strip())
        if not m:
            continue
        key = m.group(1)
        author = _field(entry, "author")
        year = _field(entry, "year")
        first = author.split(" and ")[0]
        surname = first.split(",")[0].strip() if "," in first else first.split()[-1]
        surname = re.sub(r"[{}\\]", "", surname).strip().lower()
        if surname and year:
            out[(surname, year)] = key
            # "Coello Coello" is cited as "Coello Coello et al." — index the last word too.
            if " " in surname:
                out[(surname.split()[-1], year)] = key
                out[(surname.split()[0], year)] = key
    return out


def _field(entry: str, name: str) -> str:
    m = re.search(rf"\b{name}\s*=\s*", entry)
    if not m:
        return ""
    i = m.end()
    if entry[i] != "{":
        j = entry.find(",", i)
        return entry[i:j].strip().strip('"')
    depth = 0
    for j in range(i, len(entry)):
        if entry[j] == "{":
            depth += 1
        elif entry[j] == "}":
            depth -= 1
            if depth == 0:
                return re.sub(r"[{}]", "", entry[i + 1:j]).strip()
    return ""


class Builder:
    def __init__(self, cites: dict[tuple[str, str], str]):
        self.cites = cites
        self.unresolved: list[str] = []

    # ---------------- citations ----------------
    def _key(self, surname: str, year: str) -> str | None:
        name = surname.strip().lower()
        for probe in (name, name.split()[-1] if " " in name else name,
                      name.split()[0] if " " in name else name):
            k = self.cites.get((probe, year))
            if k:
                return k
        return None

    def _parenthetical(self, m: re.Match) -> str:
        """Convert a parenthetical citation group.

        Some parentheses mix a citation with ordinary text — "(Qwen2.5-3B-Instruct; Qwen Team,
        2024; greedy decoding)" — so each semicolon-separated part is judged on its own and only
        the citation parts become \\cite. If nothing in the group is a citation, the parenthesis is
        left exactly as written.
        """
        parts_out, keys, any_cite = [], [], False
        for part in m.group(1).split(";"):
            part = part.strip()
            pm = re.match(r"([A-Z][\w\-'’]+(?:\s+[A-Z][\w\-'’]+)?)\s*(?:et al\.|&\s*[A-Z][\w\-'’]+)?,?\s*(\d{4})$", part)
            if pm:
                k = self._key(pm.group(1), pm.group(2))
                if k:
                    any_cite = True
                    keys.append(k)
                    continue
                self.unresolved.append(part)
            parts_out.append(part)
        if not any_cite:
            return m.group(0)
        cite = "\\cite{" + ",".join(keys) + "}"
        if not parts_out:
            return cite
        return "(" + "; ".join(parts_out) + " " + cite + ")"

    def _narrative(self, m: re.Match) -> str:
        name, year = m.group(1), m.group(2)
        k = self._key(name.split()[0] if " " in name else name, year)
        if not k:
            k = self._key(name, year)
        if not k:
            self.unresolved.append(f"{name} ({year})")
            return m.group(0)
        return f"{m.group(0).split('(')[0].strip()}~\\cite{{{k}}}"

    def citations(self, text: str) -> str:
        # Narrative first: "Gershon et al. (2025)" / "Kaur et al. (2023) and Khalid et al. (2026)"
        text = re.sub(r"([A-Z][\w\-'’]+(?:\s+[A-Z][\w\-'’]+)?)\s+et al\.\s*\((\d{4})\)",
                      self._narrative, text)
        text = re.sub(r"([A-Z][\w\-'’]+)\s+&\s+[A-Z][\w\-'’]+\s*\((\d{4})\)", self._narrative, text)
        # Then parenthetical groups.
        # Allow text after the year inside the parenthesis ("(Qwen2.5-3B-Instruct; Qwen Team,
        # 2024; greedy decoding)"); _parenthetical returns the original when no part is a citation.
        return re.sub(r"\(([^()]*\d{4}[a-z]?[^()]*)\)", self._parenthetical, text)

    # ---------------- inline markup ----------------
    def inline(self, text: str) -> str:
        text = self.citations(text)
        # Escape LaTeX specials in plain text, protecting the commands we just inserted.
        placeholders: list[str] = []

        def stash(m):
            placeholders.append(m.group(0))
            return f"\x00{len(placeholders) - 1}\x00"

        text = re.sub(r"\\cite\{[^}]*\}|`[^`]+`", stash, text)
        for ch in "&%#":
            text = text.replace(ch, "\\" + ch)
        text = text.replace("_", "\\_").replace("$", "\\$")
        text = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", text)
        text = re.sub(r"(?<![\*\w])\*([^*]+?)\*(?![\*\w])", r"\\textit{\1}", text)
        for k, v in UNICODE.items():
            text = text.replace(k, v)

        def unstash(m):
            raw = placeholders[int(m.group(1))]
            if raw.startswith("`"):
                body = raw.strip("`").replace("\\", "\\textbackslash{}")
                for ch in "&%#_$":
                    body = body.replace(ch, "\\" + ch)
                return "\\texttt{" + body + "}"
            return raw

        return re.sub(r"\x00(\d+)\x00", unstash, text)

    # ---------------- block level ----------------
    def table(self, rows: list[list[str]]) -> str:
        ncol = len(rows[0])
        spec = "l" + "r" * (ncol - 1) if ncol > 1 else "l"
        # Five or more columns overrun the text block at 11pt; scale those to the line width
        # rather than letting a column fall off the page.
        wide = ncol >= 5
        out = ["\\begin{center}", "\\small"]
        if wide:
            out.append("\\resizebox{\\linewidth}{!}{%")
        out += [f"\\begin{{tabular}}{{{spec}}}", "\\toprule"]
        out.append(" & ".join(self.inline(c) for c in rows[0]) + " \\\\")
        out.append("\\midrule")
        for r in rows[1:]:
            cells = (r + [""] * ncol)[:ncol]
            out.append(" & ".join(self.inline(c) for c in cells) + " \\\\")
        out += ["\\bottomrule", "\\end{tabular}"]
        if wide:
            out.append("}")
        out.append("\\end{center}")
        return "\n".join(out)

    def convert(self, md: str, base_level: int = 1) -> str:
        md = strip_working_notes(md)
        lines = md.split("\n")
        out: list[str] = []
        i = 0
        list_env: str | None = None

        def close_list():
            nonlocal list_env
            if list_env:
                out.append(f"\\end{{{list_env}}}")
                list_env = None

        while i < len(lines):
            line = lines[i]

            # markdown table
            if line.strip().startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:\-|]+\|$", lines[i + 1].strip()):
                close_list()
                rows = []
                while i < len(lines) and lines[i].strip().startswith("|"):
                    if not re.match(r"^\|[\s:\-|]+\|$", lines[i].strip()):
                        rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                    i += 1
                out.append(self.table(rows))
                continue

            m = re.match(r"^(#{1,4})\s+(.*)$", line)
            if m:
                close_list()
                level = len(m.group(1))
                title = re.sub(r"^\d+(\.\d+)*[a-z]?\.?\s*", "", m.group(2)).strip()
                if level == 1:
                    i += 1
                    continue  # the file title; the section command is emitted by the caller
                cmd = {2: "subsection", 3: "subsubsection", 4: "paragraph"}.get(level, "paragraph")
                out.append(f"\\{cmd}{{{self.inline(title)}}}")
                i += 1
                continue

            m = re.match(r"^(\d+)\.\s+(.*)$", line)
            if m:
                if list_env != "enumerate":
                    close_list()
                    out.append("\\begin{enumerate}")
                    list_env = "enumerate"
                out.append("\\item " + self.inline(m.group(2)))
                i += 1
                continue

            m = re.match(r"^[-*]\s+(.*)$", line)
            if m:
                if list_env != "itemize":
                    close_list()
                    out.append("\\begin{itemize}")
                    list_env = "itemize"
                out.append("\\item " + self.inline(m.group(1)))
                i += 1
                continue

            if line.startswith("> "):
                close_list()
                out.append("\\begin{quote}\n" + self.inline(line[2:]) + "\n\\end{quote}")
                i += 1
                continue

            if not line.strip():
                close_list()
                out.append("")
                i += 1
                continue

            if list_env and line.startswith("   "):
                out[-1] += " " + self.inline(line.strip())
                i += 1
                continue

            # Accumulate the whole paragraph before converting: citations are frequently split
            # across a line break ("Barreiros et al.,\n2025)"), and a per-line regex would pass
            # them through silently — which is how three of them slipped into an earlier build.
            close_list()
            para = [line]
            i += 1
            while i < len(lines) and lines[i].strip() and not re.match(
                    r"^(#{1,4}\s|\d+\.\s|[-*]\s|>\s|\|)", lines[i]):
                para.append(lines[i])
                i += 1
            out.append(self.inline(" ".join(x.strip() for x in para)))

        close_list()
        return "\n".join(out)


def read_generated_tables(tables_dir: Path) -> list[tuple[str, list[list[str]]]]:
    """Generated result tables, in manuscript order, from their CSVs."""
    wanted = [
        ("table_main_comparison", "Table 1. Component decomposition on MIMIC-IV Top-50 (n = 17,151)"),
        ("table2_reference_points", "Table 2. Reference points: note-blind floor and positive controls"),
        ("table4_capacity_ablation", "Table 4. Capacity ablation: Qwen2.5-3B vs 7B (paired, 200 notes)"),
        ("table5_scalability", "Table 5. Scalability across label spaces"),
        ("table6_steelman", "Table 6. Steelman 2 x 2: scorer context crossed with model size"),
    ]
    out = []
    for stem, caption in wanted:
        p = tables_dir / f"{stem}.csv"
        if not p.exists():
            print(f"  note: {p.name} absent, table omitted from the build")
            continue
        with open(p, encoding="utf-8") as fh:
            rows = [r for r in csv.reader(fh) if r]
        out.append((caption, rows))
    return out


# ---------------------------------------------------------------------------
# Minimal DOCX writer
#
# Journals ask for .docx and this environment has neither pandoc nor python-docx (and installing
# is out of scope), so the Word file is written directly: a .docx is a zip of OOXML parts, and the
# subset needed for a manuscript — headings, paragraphs, bold/italic runs, and simple tables — is
# small enough to emit honestly rather than approximate with RTF.
# ---------------------------------------------------------------------------
import zipfile
from xml.sax.saxutils import escape as _xesc

W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'


def _runs(text: str) -> str:
    """Split **bold** / *italic* / `code` into Word runs."""
    out, pos = [], 0
    for m in re.finditer(r"\*\*(.+?)\*\*|(?<![\*\w])\*([^*]+?)\*(?![\*\w])|`([^`]+)`", text):
        if m.start() > pos:
            out.append(("", text[pos:m.start()]))
        if m.group(1) is not None:
            out.append(("b", m.group(1)))
        elif m.group(2) is not None:
            out.append(("i", m.group(2)))
        else:
            out.append(("c", m.group(3)))
        pos = m.end()
    if pos < len(text):
        out.append(("", text[pos:]))
    xml = []
    for kind, body in out:
        if not body:
            continue
        props = {"b": "<w:b/>", "i": "<w:i/>", "c": '<w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/>'}.get(kind, "")
        rpr = f"<w:rPr>{props}</w:rPr>" if props else ""
        xml.append(f'<w:r>{rpr}<w:t xml:space="preserve">{_xesc(body)}</w:t></w:r>')
    return "".join(xml)


def _para(text: str, style: str = "") -> str:
    ppr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    return f"<w:p>{ppr}{_runs(text)}</w:p>"


def _table_xml(rows: list[list[str]]) -> str:
    out = ['<w:tbl><w:tblPr><w:tblBorders>'
           + "".join(f'<w:{e} w:val="single" w:sz="4" w:color="999999"/>'
                     for e in ("top", "left", "bottom", "right", "insideH", "insideV"))
           + "</w:tblBorders></w:tblPr>"]
    for r_i, row in enumerate(rows):
        cells = []
        for cell in row:
            body = f"**{cell}**" if r_i == 0 and not cell.startswith("**") else cell
            cells.append(f"<w:tc><w:tcPr/>{_para(body)}</w:tc>")
        out.append("<w:tr>" + "".join(cells) + "</w:tr>")
    out.append("</w:tbl>" + _para(""))
    return "".join(out)


def md_to_docx_body(md: str) -> str:
    lines, out, i = md.split("\n"), [], 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:\-|]+\|$", lines[i + 1].strip()):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                if not re.match(r"^\|[\s:\-|]+\|$", lines[i].strip()):
                    rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            out.append(_table_xml(rows))
            continue
        m = re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            out.append(_para(m.group(2), f"Heading{min(len(m.group(1)), 4)}"))
            i += 1
            continue
        if not line.strip():
            i += 1
            continue
        para = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not re.match(r"^(#{1,4}\s|\|)", lines[i]):
            para.append(lines[i])
            i += 1
        out.append(_para(" ".join(x.strip() for x in para)))
    return "".join(out)


def write_docx(md: str, path: Path) -> None:
    doc = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document {W}><w:body>'
           + md_to_docx_body(md) + "</w:body></w:document>")
    styles = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:styles {W}>'
              + "".join(
                  f'<w:style w:type="paragraph" w:styleId="Heading{n}"><w:name w:val="heading {n}"/>'
                  f'<w:pPr><w:outlineLvl w:val="{n - 1}"/><w:spacing w:before="240" w:after="120"/></w:pPr>'
                  f'<w:rPr><w:b/><w:sz w:val="{32 - 4 * n}"/></w:rPr></w:style>' for n in (1, 2, 3, 4))
              + "</w:styles>")
    ct = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
          '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
          '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
          '<Default Extension="xml" ContentType="application/xml"/>'
          '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
          '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
          "</Types>")
    rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
            "</Relationships>")
    drels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
             '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
             '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
             "</Relationships>")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", ct)
        z.writestr("_rels/.rels", rels)
        z.writestr("word/document.xml", doc)
        z.writestr("word/styles.xml", styles)
        z.writestr("word/_rels/document.xml.rels", drels)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manuscript", default="manuscript")
    ap.add_argument("--out", default="manuscript/build")
    args = ap.parse_args()
    M, OUT = Path(args.manuscript), Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)

    bib = M / "references" / "references.bib"
    builder = Builder(load_citation_map(bib))

    # ---------- Markdown assembly ----------
    md_parts = [(M / "sections" / "00_title_page.md").read_text(encoding="utf-8"),
                (M / "sections" / "01_abstract.md").read_text(encoding="utf-8")]
    for fname, _ in SECTIONS:
        md_parts.append((M / "sections" / fname).read_text(encoding="utf-8"))
    for extra in ("09_declarations.md", "09_acknowledgements.md"):
        p = M / "sections" / extra
        if p.exists():
            md_parts.append(p.read_text(encoding="utf-8"))
    caps = M / "figures" / "FIGURE_CAPTIONS.md"
    if caps.exists():
        md_parts.append(caps.read_text(encoding="utf-8"))
    md_path = OUT / "MORAG-ICD_manuscript.md"
    md_path.write_text(strip_working_notes("\n\n---\n\n".join(md_parts)), encoding="utf-8")
    print(f"  wrote {md_path} ({len(md_path.read_text(encoding='utf-8').split())} words incl. front matter)")

    # ---------- LaTeX ----------
    abstract_md = (M / "sections" / "01_abstract.md").read_text(encoding="utf-8")
    body_lines = [l for l in abstract_md.split("\n") if l.strip() and not l.startswith("#")]
    kw_idx = next((i for i, l in enumerate(body_lines) if l.lower().startswith("**keywords")), len(body_lines))
    abstract_tex = builder.convert("\n".join(body_lines[:kw_idx]))
    keywords = re.sub(r"^\*\*Keywords:\*\*\s*", "", " ".join(body_lines[kw_idx:])).strip()

    tex = [
        # pdftex.def auto-loads epstopdf-base whenever shell-escape is on; this TeX Live install
        # does not ship it, and every figure here is already a PDF. The documented opt-out is to
        # define this before \\documentclass (see graphics-def/pdftex.def).
        "\\newcommand{\\DoNotLoadEpstopdf}{}",
        "\\documentclass[11pt]{article}",
        "\\usepackage[utf8]{inputenc}",
        # lmodern before fontenc: this install has no cm-super Type1 fonts, so T1 with the
        # default Computer Modern fails at the PDF-writing stage; Latin Modern ships T1 outlines.
        "\\usepackage{lmodern}",
        "\\usepackage[T1]{fontenc}",
        "\\usepackage[margin=2.5cm]{geometry}",
        "\\usepackage{graphicx}",
        "\\usepackage{booktabs}",
        "\\usepackage{amsmath}",
        "\\usepackage{url}",
        # The manuscript's table numbers are fixed by the prose (Table 3 and Table M1 are inline,
        # so LaTeX's own counter would renumber the floats and break every cross-reference).
        # Captions therefore carry their number literally and LaTeX contributes no label.
        "\\usepackage{caption}",
        "\\captionsetup[table]{labelformat=empty}",
        # Same for figures: the supplementary figure is "Figure S1" in the text, which LaTeX's
        # counter would rewrite as "Figure 5".
        "\\captionsetup[figure]{labelformat=empty}",
        "\\usepackage{setspace}\\onehalfspacing",
        "\\usepackage[hidelinks]{hyperref}",
        "\\title{Why a Retrieval-Augmented LLM Loses to TF-IDF at ICD-10 Coding:\\\\"
        "A Component-Wise Cautionary Study}",
        r'\author{Muhammed Yusuf K\"u\c{c}\"ukkara \\[3pt]' "\n"
        r'\small Department of Computer Engineering, Faculty of Technology, \\' "\n"
        r'\small Sakarya University of Applied Sciences, Sakarya, T\"urkiye \\[3pt]' "\n"
        r'\small ORCID: 0000-0003-0600-3651 \quad \texttt{muhammedkucukkara@subu.edu.tr}}',
        "\\date{}",
        "\\begin{document}",
        "\\maketitle",
        "\\begin{abstract}", abstract_tex, "\\end{abstract}",
        "\\noindent\\textbf{Keywords:} " + builder.inline(keywords), "",
    ]

    for fname, title in SECTIONS:
        md = (M / "sections" / fname).read_text(encoding="utf-8")
        tex.append(f"\\section{{{title}}}")
        tex.append(builder.convert(md))
        tex.append("")

    tables = read_generated_tables(M / "tables" / "generated" / "tables" / "top50")
    if tables:
        tex.append("\\clearpage\n\\section*{Tables}")
        for caption, rows in tables:
            tex.append("\\begin{table}[htbp]\\centering")
            tex.append("\\caption{" + builder.inline(caption) + "}")
            tex.append(builder.table(rows))
            tex.append("\\end{table}")
        tex.append("")

    figs = [("F1_pipeline", "Figure 1"), ("F2_decomposition_ladder", "Figure 2"),
            ("F3_judgement_vs_outcome", "Figure 3"), ("F4_evidence_leakage", "Figure 4"),
            ("S1_scalability", "Figure S1")]
    cap_text = caps.read_text(encoding="utf-8") if caps.exists() else ""
    present = [(stem, label) for stem, label in figs if (M / "figures" / f"{stem}.pdf").exists()]
    if present:
        tex.append("\\clearpage\n\\section*{Figures}")
        for stem, label in present:
            m = re.search(rf"\*\*{label}\.\s*(.+?)\*\*(.*?)(?=\n\*\*Figure|\Z)", cap_text, re.S)
            caption = f"{label}. " + (m.group(1) + " " + m.group(2)).strip() if m else label
            caption = re.sub(r"\s+", " ", caption)
            tex.append("\\begin{figure}[htbp]\\centering")
            tex.append(f"\\includegraphics[width=\\linewidth]{{../figures/{stem}.pdf}}")
            tex.append("\\caption{" + builder.inline(caption) + "}")
            tex.append("\\end{figure}")
        tex.append("")

    for extra, heading in (("09_acknowledgements.md", "Acknowledgements"),
                           ("09_declarations.md", "Declarations")):
        p = M / "sections" / extra
        if p.exists():
            tex.append(f"\\section*{{{heading}}}" if extra.startswith("09_ack")
                       else f"\\clearpage\n\\section*{{{heading}}}")
            tex.append(builder.convert(p.read_text(encoding="utf-8")))

    tex += ["\\clearpage", "\\bibliographystyle{unsrt}", "\\bibliography{references}",
            "\\end{document}"]

    (OUT / "main.tex").write_text("\n".join(tex), encoding="utf-8")
    shutil.copy(bib, OUT / "references.bib")
    print(f"  wrote {OUT / 'main.tex'} and references.bib")

    docx_path = OUT / "MORAG-ICD_manuscript.docx"
    write_docx(md_path.read_text(encoding="utf-8"), docx_path)
    print(f"  wrote {docx_path}")

    # A citation that the regexes never matched would pass through as plain text rather than
    # raising, so scan the generated LaTeX for anything still shaped like an author-year citation.
    tex_text = (OUT / "main.tex").read_text(encoding="utf-8")
    residual = sorted(set(
        m.group(0) for m in re.finditer(r"[A-Z][\w\-']+(?:\s+et al\.|\s*&\s*[A-Z][\w\-']+)[,]?\s*\(?\d{4}\)?", tex_text)
        if "\\cite" not in tex_text[max(0, m.start() - 12):m.end() + 12]))
    if residual:
        print(f"\nERROR: {len(residual)} author-year citation(s) survived conversion to LaTeX:")
        for r in residual:
            print(f"  - {r}")
        return 1

    if builder.unresolved:
        uniq = sorted(set(builder.unresolved))
        print(f"\nERROR: {len(uniq)} citation(s) could not be resolved to a bib key:")
        for u in uniq:
            print(f"  - {u}")
        return 1
    print("\nall in-text citations resolved to bib keys")
    return 0


if __name__ == "__main__":
    sys.exit(main())
