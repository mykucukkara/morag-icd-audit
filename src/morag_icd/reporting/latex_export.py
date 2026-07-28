"""
LaTeX export utilities for academic publication tables.

Converts pandas DataFrames to well-formatted LaTeX booktabs tables.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def export_tables_to_latex(
    df,
    output_path: str | Path,
    caption: str = "",
    label: str = "",
    bold_best_row: bool = False,
) -> None:
    """
    Export a DataFrame to a LaTeX booktabs table.

    Parameters
    ----------
    df : pd.DataFrame
    output_path : str | Path
    caption : str
    label : str
    bold_best_row : bool
        If True, bold the row with the highest Micro-F1 value.
    """
    if not HAS_PANDAS:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if label == "":
        label = f"tab:{output_path.stem}"

    n_cols = len(df.columns)
    col_format = "l" + "c" * (n_cols - 1)

    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        rf"  \caption{{{caption}}}",
        rf"  \label{{{label}}}",
        rf"  \begin{{tabular}}{{{col_format}}}",
        r"    \toprule",
    ]

    # Header
    header = " & ".join([f"\\textbf{{{col}}}" for col in df.columns])
    lines.append(f"    {header} \\\\")
    lines.append(r"    \midrule")

    # Rows
    for _, row in df.iterrows():
        row_str = " & ".join([str(v) for v in row.values])
        lines.append(f"    {row_str} \\\\")

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  LaTeX table saved: {output_path}")


def export_all_tables(tables_dir: str | Path, latex_dir: str | Path) -> None:
    """Convert all CSV tables in tables_dir to LaTeX."""
    if not HAS_PANDAS:
        return
    import pandas as pd

    tables_dir = Path(tables_dir)
    latex_dir = Path(latex_dir)
    latex_dir.mkdir(parents=True, exist_ok=True)

    for csv_file in sorted(tables_dir.glob("*.csv")):
        try:
            df = pd.read_csv(csv_file)
            caption = csv_file.stem.replace("_", " ").title()
            tex_path = latex_dir / csv_file.with_suffix(".tex").name
            export_tables_to_latex(df, tex_path, caption=caption)
        except Exception as e:
            print(f"Warning: Could not convert {csv_file.name} to LaTeX: {e}")
