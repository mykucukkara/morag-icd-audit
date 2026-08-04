#!/usr/bin/env python3
"""
Script 46: build the Journal of Biomedical Informatics submission package.

JBI is an Elsevier journal, so the parts an initial submission needs are Elsevier's, not this
project's: a title page carrying the author details separately from the blinded-ish manuscript,
Highlights (three to five bullets, 85 characters each at most — a hard limit their system enforces),
a declaration of interest, a CRediT statement, and numbered references.

The class is deliberately `article` rather than `elsarticle`. Elsevier's "Your Paper Your Way"
policy accepts any readable format at initial submission and only requires their template at
revision; `elsarticle.cls` is not installed here and downloading it is out of scope. Switching later
is a one-line change, documented in the generated README.

Reuses scripts/45 for the Markdown-to-LaTeX conversion and citation resolution, so the JBI build
cannot drift from the main build.

Usage:
    python scripts/46_build_jbi_submission.py --manuscript manuscript --out manuscript/submission/jbi
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load_45():
    spec = importlib.util.spec_from_file_location("build45", HERE / "45_build_manuscript.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


B45 = _load_45()

# Elsevier caps each highlight at 85 characters including spaces. Written to say what was found,
# not what was done, because that is what a highlight is for.
def selected_title(sections_dir: Path) -> str:
    """Read the chosen title from the title page.

    It was hard-coded in three places here — cover letter, title page, CAS front matter — which is
    how the package came to advertise a superseded title after the manuscript's was revised. One
    source, and a hard failure if it is missing.
    """
    text = (sections_dir / "00_title_page.md").read_text(encoding="utf-8")
    m = re.search(r'Selected:\s*\*\*"?(.+?)"?\*\*', text, re.S)
    if not m:
        raise SystemExit("00_title_page.md has no `Selected: **...**` line — cannot build")
    return " ".join(m.group(1).split())


HIGHLIGHTS = [
    #  Was "neither factor helps on its own" — the claim §4.5 retracted: capacity alone is +0.036
    #  with a CI excluding zero. A highlight carries no surrounding text to qualify it, so it says
    #  the two simple effects instead of asserting a joint null. `scripts/44` now scans this list.
    "Model capacity is worth +0.036 without the note and +0.320 with it: they interact",
    "The same pipeline fails a note-blind floor at 3B and clears it by 0.130 at 14B",
    "Single-factor evaluations mistake a configuration limit for an architectural one",
    "An oracle over the same shortlist scores twice the model: the loss is in selection",
    "Self-reported evidence and confidence fields are unvalidated; we give an audit list",
]

COVER_LETTER = """\
{date}

The Editors
Journal of Biomedical Informatics

Dear Editors,

I submit for your consideration "{title}".

The paper reports a negative result and, more importantly, the protocol that makes it credible. I
decomposed a complete evidence-constrained, contrastive RAG-LLM pipeline for ICD-10 coding of
MIMIC-IV discharge summaries into a fourteen-system ladder and evaluated every arm on the same
subject-disjoint split with note-level paired significance testing. The full pipeline reaches
micro-F1 0.133 against 0.449 for a tuned TF-IDF baseline, and in the ladder's as-run configuration
(3B, note withheld) every retrieval, RAG and full-model arm falls significantly below a constant
predictor that never reads the note. Measured against the
arm each component is added to, the damage is localized: the evidence filter costs 0.053 micro-F1
and discards two thirds of the gold codes retrieval had retained, while the contrastive verifier is
mildly beneficial on its own. At matched output cardinality an oracle over the same shortlist
reaches 0.264 against the model's 0.133 and a random null's 0.119, so the selector captures about a
tenth of what is recoverable.

Two findings go beyond this one system. First, scorer context and model size interact, and the
interaction decides the headline. Across 3B, 7B and 14B measured on one note set, enlarging the
scorer gains 0.036 micro-F1 when the note is withheld and 0.320 when it is supplied; the interaction
is +0.284 (95% CI +0.263 to +0.305) and is significant at every step. This matters because the two
cells sit on opposite sides of the note-blind floor: starved of context at 3B the pipeline falls
significantly below a predictor that never reads the note, while at 14B with truncated context it
clears that floor by 0.130 (95% CI +0.115 to +0.145) and comes within 0.024 of tuned TF-IDF, still
significantly behind. An evaluation varying either factor alone would have measured a real but small
effect — +0.036 for capacity, −0.033 for context at 3B — and so would have substantially understated
the joint effect; varying one factor at a time is what the two published designs closest to ours do.
Second, the pipeline's
self-reported evidence signals do not carry the information they appear to: an unsupported rate that
is the arithmetic complement of the support rate, a support rate of 1.000 in the filtered arms that
is a property of the filter rather than a measurement, a confidence field whose omission silently
corrupted top-k selection, and an exact-quote compliance rate of 8-21% against the passage the model
was shown — a rate that falls rather than rises when the prompt states that the field is checked by
exact string matching.

I believe this fits JBI's methodological line rather than a systems line. The contribution readers
can reuse is the audit protocol — a note-blind floor, an oracle-over-shortlist decomposition,
note-level paired testing, an evidence-provenance measurement, and a grounding metric the model
cannot self-report — together with a reporting checklist for evidence-grounded coding claims. The pipeline
and the audit code are both public ({repo}) and archived ({doi}), including the script that
re-checks every number in the manuscript against its stored artifact; it caught two errors in my own
reporting, which is the argument for the practice.

I should state the study's main limitation plainly. I could not reproduce a PLM-ICD-class supervised
system (micro-F1 approximately 0.70) because it requires a clinically pretrained encoder unavailable
in the offline environment. My strongest supervised control is a label-wise attention coder from
that architecture family, trained on the same split with a general-domain encoder, which reaches
0.559; that bounds the missing control rather than closing it. The comparison should
therefore be read as a decomposition of one pipeline against competent baselines, not as a claim
about the ceiling of supervised coding. This is stated in Section 4.1a and in Limitation 1.

The work is single-authored and has not been published elsewhere or submitted concurrently. MIMIC-IV
was used under the PhysioNet Credentialed Health Data Use Agreement; no clinical text or identifier
appears in the manuscript, the figures, or the released artifacts. I declare no competing interests.

Thank you for your time.

Yours sincerely,

Muhammed Yusuf Küçükkara
Department of Computer Engineering, Faculty of Technology
Sakarya University of Applied Sciences, Sakarya, Türkiye
muhammedkucukkara@subu.edu.tr
ORCID 0000-0003-0600-3651
"""


def build_title_page(builder) -> str:
    return r"""\newcommand{\DoNotLoadEpstopdf}{}
\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage[T1]{fontenc}
\usepackage[margin=2.5cm]{geometry}
\usepackage[hidelinks]{hyperref}
\pagestyle{empty}
\begin{document}

\begin{center}
{\Large\bfseries TITLE_PLACEHOLDER}
\end{center}

\vspace{1.5em}
\noindent\textbf{Author}\\[4pt]
Muhammed Yusuf K\"u\c{c}\"ukkara\\
Department of Computer Engineering, Faculty of Technology\\
Sakarya University of Applied Sciences, Sakarya, T\"urkiye\\
ORCID: \href{https://orcid.org/0000-0003-0600-3651}{0000-0003-0600-3651}

\vspace{1.5em}
\noindent\textbf{Corresponding author}\\[4pt]
Muhammed Yusuf K\"u\c{c}\"ukkara\\
\href{mailto:muhammedkucukkara@subu.edu.tr}{muhammedkucukkara@subu.edu.tr}

\vspace{1.5em}
\noindent\textbf{Running title}\\[4pt]
Decomposing RAG-LLM ICD-10 coding

\vspace{1.5em}
\noindent\textbf{Declaration of interest}\\[4pt]
None. The author declares no competing financial interests or personal relationships that could have
appeared to influence the work reported in this paper.

\vspace{1.5em}
\noindent\textbf{Funding}\\[4pt]
This research received no specific grant from any funding agency in the public, commercial, or
not-for-profit sectors.

\vspace{1.5em}
\noindent\textbf{CRediT author statement}\\[4pt]
\textbf{Muhammed Yusuf K\"u\c{c}\"ukkara:} Conceptualization, Methodology, Software, Investigation,
Formal analysis, Data curation, Writing -- Original draft, Writing -- Review \& editing,
Visualization.

\end{document}
"""


def build_highlights(builder) -> str:
    items = "\n".join(r"\item " + builder.inline(h) for h in HIGHLIGHTS)
    return r"""\newcommand{\DoNotLoadEpstopdf}{}
\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage[T1]{fontenc}
\usepackage[margin=2.5cm]{geometry}
\pagestyle{empty}
\begin{document}
\begin{center}{\large\bfseries Highlights}\end{center}
\vspace{1em}
\begin{itemize}
""" + items + r"""
\end{itemize}
\end{document}
"""


CAS_REQUIRED = ["balance", "charis", "inconsolata", "l3regex", "makecell", "moreverb",
                "multirow", "natbib", "stfloats", "stix", "wrapfig", "xstring"]


def cas_available(template_dir: Path) -> tuple[bool, list[str]]:
    """Can Elsevier's CAS class actually compile here?

    The class itself is only the entry point; it pulls in a dozen packages and three font
    families. Reporting exactly which are missing is more useful than a compile failure, because
    the fix is per-package.
    """
    if not (template_dir / "cas-sc.cls").exists():
        return False, ["cas-sc.cls (template folder absent)"]
    missing = []
    for pkg in CAS_REQUIRED:
        if (template_dir / f"{pkg}.sty").exists():
            continue
        r = subprocess.run(["kpsewhich", f"{pkg}.sty"], capture_output=True, text=True)
        if not r.stdout.strip():
            missing.append(pkg)
    return (not missing), missing


def build_cas(M: Path, OUT: Path, builder, template_dir: Path, title_tex: str) -> None:
    """Elsevier CAS single-column version, front matter in the class's own commands."""
    for f in ("cas-sc.cls", "cas-common.sty", "cas-model2-names.bst"):
        src = template_dir / f
        if src.exists():
            shutil.copy(src, OUT / f)

    body_main = (M / "build" / "main.tex").read_text(encoding="utf-8")
    body = body_main[body_main.index("\\section{Introduction}"):body_main.index("\\bibliographystyle")]
    body = body.replace("{../figures/", "{figures/")

    abstract_md = (M / "sections" / "01_abstract.md").read_text(encoding="utf-8")
    lines = [l for l in abstract_md.split("\n") if l.strip() and not l.startswith("#")]
    kw_idx = next((i for i, l in enumerate(lines) if l.lower().startswith("**keywords")), len(lines))
    abstract = builder.convert("\n".join(lines[:kw_idx]))
    keywords = re.sub(r"^\*\*Keywords:\*\*\s*", "", " ".join(lines[kw_idx:])).strip()
    kw_sep = " \\sep\n".join(k.strip() for k in keywords.split(";") if k.strip())
    highlights = "\n".join(r"\item " + builder.inline(h) for h in HIGHLIGHTS)

    tex = r"""\documentclass[a4paper,fleqn]{cas-sc}
\usepackage[numbers]{natbib}
\begin{document}
\let\WriteBookmarks\relax
\def\floatpagepagefraction{1}
\def\textpagefraction{.001}

\shorttitle{Decomposing RAG-LLM ICD-10 coding}
\shortauthors{M. Y. K\"u\c{c}\"ukkara}

\title[mode = title]{TITLE_PLACEHOLDER}

\author[1]{Muhammed Yusuf K\"u\c{c}\"ukkara}[orcid=0000-0003-0600-3651]
\cormark[1]
\ead{muhammedkucukkara@subu.edu.tr}
\credit{Conceptualization, Methodology, Software, Investigation, Formal analysis, Data curation,
Writing -- original draft, Writing -- review \& editing, Visualization}

\affiliation[1]{organization={Department of Computer Engineering, Faculty of Technology,
                              Sakarya University of Applied Sciences},
                city={Sakarya},
                country={T\"urkiye}}

\cortext[1]{Corresponding author.}

\begin{abstract}
""" + abstract + r"""
\end{abstract}

\begin{highlights}
""" + highlights + r"""
\end{highlights}

\begin{keywords}
""" + kw_sep + r"""
\end{keywords}

\maketitle

""" + body + r"""
\printcredits

\bibliographystyle{cas-model2-names}
\bibliography{references}

\end{document}
"""
    (OUT / "manuscript_cas.tex").write_text(
        tex.replace("TITLE_PLACEHOLDER", title_tex), encoding="utf-8")
    print("  wrote manuscript_cas.tex (Elsevier CAS single-column)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manuscript", default="manuscript")
    ap.add_argument("--out", default="manuscript/submission/jbi")
    ap.add_argument("--date", default="", help="date for the cover letter; blank leaves a placeholder")
    args = ap.parse_args()
    M, OUT = Path(args.manuscript), Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    title = selected_title(M / "sections")
    title_tex = title.replace('&', r'\&').replace('%', r'\%').replace('_', r'\_')

    over = [(h, len(h)) for h in HIGHLIGHTS if len(h) > 85]
    if over:
        print("ERROR: Elsevier caps highlights at 85 characters:")
        for h, n in over:
            print(f"  {n} chars: {h}")
        return 1
    if not 3 <= len(HIGHLIGHTS) <= 5:
        print(f"ERROR: Elsevier wants 3-5 highlights, got {len(HIGHLIGHTS)}")
        return 1

    builder = B45.Builder(B45.load_citation_map(M / "references" / "references.bib"))

    (OUT / "title_page.tex").write_text(
        build_title_page(builder).replace("TITLE_PLACEHOLDER", title_tex), encoding="utf-8")
    (OUT / "highlights.tex").write_text(build_highlights(builder), encoding="utf-8")
    print(f"  wrote title_page.tex and highlights.tex "
          f"({len(HIGHLIGHTS)} highlights, longest {max(len(h) for h in HIGHLIGHTS)} chars)")

    # ---- the manuscript itself: same content, JBI presentation ----
    main_tex = (M / "build" / "main.tex").read_text(encoding="utf-8")
    if "\\begin{document}" not in main_tex:
        print("ERROR: run scripts/45 first; manuscript/build/main.tex is missing or malformed")
        return 1

    # Double spacing for review, and the author block moves to the separate title page.
    jbi = main_tex.replace("\\usepackage{setspace}\\onehalfspacing",
                           "\\usepackage{setspace}\\doublespacing")
    jbi = re.sub(r"\\author\{.*?\}\n(?=\\date)", "\\\\author{}\n", jbi, flags=re.S)
    jbi = jbi.replace("\\maketitle", "\\maketitle\n\\thispagestyle{empty}")
    # Figures are copied next to this file, not one level up as in manuscript/build/.
    jbi = jbi.replace("{../figures/", "{figures/")
    (OUT / "manuscript.tex").write_text(jbi, encoding="utf-8")
    shutil.copy(M / "references" / "references.bib", OUT / "references.bib")
    print("  wrote manuscript.tex (double-spaced, author block moved to the title page)")

    date = args.date or "[date]"
    #  The repository URL and the DOIs were hard-coded here, which is how the cover letter came to
    #  advertise v1.2.0 after the manuscript had moved on — the same failure the title had, for the
    #  same reason. They now come from manuscript/release.json, which scripts/44 checks the two
    #  manuscript sections against, so all three sites are one source.
    #  Fails with a sentence rather than a traceback, the way the title read does. A build script
    #  that dies on a raw FileNotFoundError reads as broken; this one is merely being told that the
    #  identifiers it must not invent are missing.
    rel_path = M / "release.json"
    if not rel_path.exists():
        raise SystemExit(f"{rel_path} is missing — the cover letter's repository URL and DOIs come "
                         "from it and are deliberately not hard-coded here; cannot build")
    rel = json.loads(rel_path.read_text(encoding="utf-8"))
    cited = rel.get("cited_release") or {}
    if not cited.get("version") or not cited.get("version_doi"):
        raise SystemExit(f"{rel_path} has no cited_release.version/version_doi — publish the release "
                         "and record the DOI before building a cover letter that claims one")
    (OUT / "cover_letter.md").write_text(
        COVER_LETTER.format(date=date, title=title,
                            repo=rel["repository_url"],
                            doi=f"https://doi.org/{rel['concept_doi']} "
                                f"({cited['version']}: https://doi.org/{cited['version_doi']})"),
        encoding="utf-8")
    print("  wrote cover_letter.md")

    figs = sorted((M / "figures").glob("F*.pdf")) + sorted((M / "figures").glob("S*.pdf"))
    fig_dir = OUT / "figures"
    fig_dir.mkdir(exist_ok=True)
    for f in figs:
        shutil.copy(f, fig_dir / f.name)
    print(f"  copied {len(figs)} figures for separate upload")

    tpl = M / "els-cas-templates"
    ok, missing = cas_available(tpl)
    if ok:
        build_cas(M, OUT, builder, tpl, title_tex)
    else:
        print(f"  CAS build skipped; missing: {', '.join(missing)}")

    (OUT / "README.md").write_text(build_readme(len(figs), ok, missing), encoding="utf-8")

    # What gets uploaded to the journal is the PDF, and the PDFs are not regenerated on every
    # run (the CAS build is usually skipped for missing packages), so a package can ship sources
    # for one version of the paper and PDFs for another — this one did, for two days.
    #
    # The check is on content, not timestamps: this script rewrites the .tex files every run, so
    # any mtime comparison would fire every time and be ignored within a week. Instead, read the
    # title out of the .tex just written and confirm the PDF beside it actually contains that
    # title.
    stale = []
    title_m = re.search(r"\\Large\\bfseries\s+(.+?)\}", (OUT / "title_page.tex").read_text(encoding="utf-8"), re.S)
    if title_m and shutil.which("pdftotext"):
        # the title is line-broken in the source; compare on the first few words, which is enough
        # to tell one title from another and survives the \\[4pt] break
        probe = " ".join(re.sub(r"\\\\\[[^\]]*\]|\\\\", " ", title_m.group(1)).split())[:40]
        for pdf in sorted(OUT.glob("*.pdf")):
            if pdf.name == "highlights.pdf":
                continue            # highlights carry no title
            try:
                text = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True,
                                      text=True, timeout=120).stdout
            except Exception:
                continue
            if probe and " ".join(text.split()).find(probe) < 0:
                stale.append(pdf.name)
    if stale:
        print("\n  !! PDFs do not match the sources just written: " + ", ".join(stale))
        print(f"     none of them contains the current title ({probe!r}...).")
        print("     Rebuild them, or delete them so the package ships sources only —")
        print("     uploading these would submit the previous version of the paper.")
    elif title_m:
        print("  PDFs present in the package carry the current title")

    print(f"\npackage in {OUT}; build the PDFs with:")
    print(f"  cd {OUT} && pdflatex title_page && pdflatex highlights && "
          "pdflatex manuscript && bibtex manuscript && pdflatex manuscript && pdflatex manuscript")
    return 0


def build_readme(n_figs: int, cas_ok: bool = False, cas_missing: list | None = None) -> str:
    if cas_ok:
        cas_note = """**Elsevier CAS version available.** `manuscript_cas.tex` uses Elsevier's own
`cas-sc` class with the CAS front matter (author macros, ORCID, affiliation block, `\\credit`
statements, `highlights` and `keywords` environments) and the `cas-model2-names` bibliography style.
Build it with `pdflatex manuscript_cas && bibtex manuscript_cas && pdflatex manuscript_cas &&
pdflatex manuscript_cas`. The `article`-based `manuscript.tex` is kept as the fallback."""
    else:
        pkgs = " ".join(cas_missing or [])
        cas_note = f"""**The class is `article`, not Elsevier's CAS.** Elsevier's "Your Paper Your Way"
accepts any readable format at initial submission and asks for their template only at revision, so
this package is submittable as it stands.

The CAS template is present in `manuscript/els-cas-templates/`, but the class needs packages this
TeX Live install does not have: `{pkgs}`. Installing them into a user tree (no root needed) makes
the CAS build appear automatically on the next run of `scripts/46`:

    tlmgr init-usertree
    tlmgr --usermode option repository \\
        https://ftp.math.utah.edu/pub/tex/historic/systems/texlive/2020/tlnet-final
    tlmgr --usermode install {pkgs}

The frozen 2020 repository matters: this is TeX Live 2020, and current CTAN mirrors refuse to serve
an older release."""

    return f"""# JBI submission package

Generated by `scripts/46_build_jbi_submission.py`. Do not edit these files: edit
`manuscript/sections/*.md`, re-run `scripts/45` then `scripts/46`, and everything here regenerates.

## Build

    pdflatex title_page
    pdflatex highlights
    pdflatex manuscript && bibtex manuscript && pdflatex manuscript && pdflatex manuscript

## What to upload where in Editorial Manager

| Item type | File |
|---|---|
| Cover Letter | `cover_letter.md` (paste as text, or export to PDF) |
| Title Page | `title_page.pdf` — author, affiliation, ORCID, corresponding author, declaration of interest, funding, CRediT |
| Highlights | `highlights.pdf` — 5 bullets, all within Elsevier's 85-character cap |
| Manuscript | `manuscript.pdf` — double-spaced, numbered references, tables and figures at the end |
| Figures | `figures/` — {n_figs} files, uploaded individually |
| Declaration of Interest | stated on the title page; repeat in the submission form if asked |

## Two things to know about the format

{cas_note}

**No line numbers.** `lineno.sty` is absent here, so `\\linenumbers` is not used. Editorial Manager
adds line numbers to the reviewer PDF it builds, so this is normally not an issue — but if the
journal asks for them in the source, add `\\usepackage{{lineno}}\\linenumbers` once the package is
available.

## Reference style

Numbered, resolved from `references.bib` by BibTeX with `unsrt`, which orders references by first
citation exactly as Elsevier's numbered style does. The formatting of individual entries differs
slightly from `elsarticle-num`; that is corrected by the class swap above, or by the publisher at
proof stage.
"""


if __name__ == "__main__":
    sys.exit(main())
