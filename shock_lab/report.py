"""Export results as an Excel workbook or a PDF report."""

from __future__ import annotations

import io
from datetime import date

import pandas as pd


def to_excel(sheets: dict[str, pd.DataFrame]) -> bytes:
    """One sheet per table, with readable column widths."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, df in sheets.items():
            sheet = name[:31]  # Excel's limit
            df.to_excel(writer, sheet_name=sheet, index=False)
            ws = writer.sheets[sheet]
            for i, col in enumerate(df.columns, start=1):
                width = max([len(str(col))] + [len(f"{v:,.2f}" if isinstance(v, float) else str(v)) for v in df[col]])
                ws.column_dimensions[ws.cell(1, i).column_letter].width = min(width + 2, 40)
    return buf.getvalue()


def _fmt(v, col: str) -> str:
    if isinstance(v, str):
        return v
    if pd.isna(v):
        return "n/a"
    if col in ("Price", "Value ($)", "Value today ($)"):
        return f"${v:,.2f}" if col == "Price" else f"${v:,.0f}"
    if "($)" in col:  # gains and losses get a sign
        return f"{'-' if v < 0 else '+'}${abs(v):,.0f}"
    if "(%)" in col:
        return f"{v:+.2f}%"
    if col == "R²":
        return f"{v:.2f}"
    return f"{v:,.2f}"


def to_pdf(
    title: str,
    settings: list[tuple[str, str]],
    summary: list[tuple[str, str]],
    tables: list[tuple[str, pd.DataFrame]],
    disclaimer: str,
) -> bytes:
    """A simple portrait report: settings, headline numbers, then each table."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    styles = getSampleStyleSheet()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, leftMargin=0.6 * inch, rightMargin=0.6 * inch,
                            topMargin=0.6 * inch, bottomMargin=0.6 * inch, title=title)
    grid = TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#263238")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f4f5")]),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cfd8dc")),
        ]
    )
    story = [
        Paragraph(title, styles["Title"]),
        Paragraph(f"Generated {date.today():%B %d, %Y}", styles["Normal"]),
        Spacer(1, 10),
    ]
    kv = Table([[k, v] for k, v in settings + summary], hAlign="LEFT", colWidths=[2.2 * inch, 4.5 * inch])
    kv.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 9), ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold")]))
    story += [kv, Spacer(1, 12)]
    for heading, df in tables:
        if df.empty:
            continue
        data = [list(map(str, df.columns))] + [[_fmt(v, c) for v, c in zip(row, df.columns)] for row in df.itertuples(index=False)]
        t = Table(data, hAlign="LEFT", repeatRows=1)
        t.setStyle(grid)
        story += [Paragraph(heading, styles["Heading3"]), t, Spacer(1, 12)]
    story.append(Paragraph(disclaimer.replace("**", ""), styles["Italic"]))
    doc.build(story)
    return buf.getvalue()
