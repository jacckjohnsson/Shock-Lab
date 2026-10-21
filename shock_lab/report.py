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


def _draw_spike(canvas, x: float, y: float, scale: float, color, width: float) -> None:
    """The seismograph logo, with (x, y) at the left end of the baseline."""
    from .brand import SPIKE_POINTS

    pts = [(x + (px - 2) * scale, y - (py - 16) * scale) for px, py in SPIKE_POINTS]
    canvas.setStrokeColor(color)
    canvas.setLineWidth(width)
    canvas.setLineJoin(1)
    canvas.setLineCap(1)
    path = canvas.beginPath()
    path.moveTo(*pts[0])
    for p in pts[1:]:
        path.lineTo(*p)
    canvas.drawPath(path, stroke=1, fill=0)


def to_pdf(
    title: str,
    settings: list[tuple[str, str]],
    summary: list[tuple[str, str]],
    tables: list[tuple[str, pd.DataFrame]],
    disclaimer: str,
) -> bytes:
    """A branded landscape report: header band, settings, headline numbers, then each table.

    Every page carries the Shock Lab header, a faint seismograph watermark and the footer disclaimer.
    """
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import landscape, letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    from .brand import AMBER, NAVY, WATERMARK_TEXT

    navy, amber = colors.HexColor(NAVY), colors.HexColor(AMBER)
    page_w, page_h = landscape(letter)

    def decorate(canvas, doc) -> None:
        canvas.saveState()
        # Header band with logo and wordmark
        canvas.setFillColor(navy)
        canvas.rect(0, page_h - 0.6 * inch, page_w, 0.6 * inch, stroke=0, fill=1)
        _draw_spike(canvas, 0.45 * inch, page_h - 0.3 * inch, 0.55, amber, 1.8)
        canvas.setFillColor(colors.white)
        canvas.setFont("Courier-Bold", 14)
        canvas.drawString(0.45 * inch + 36, page_h - 0.35 * inch, "SHOCK LAB")
        canvas.setFont("Helvetica", 9)
        canvas.drawRightString(page_w - 0.45 * inch, page_h - 0.35 * inch, f"{title} · {date.today():%B %d, %Y}")
        # Faint watermark across the middle of the page
        faint = colors.Color(navy.red, navy.green, navy.blue, alpha=0.05)
        _draw_spike(canvas, page_w / 2 - 285, page_h / 2 - 20, 9.5, faint, 14)
        # Footer
        canvas.setStrokeColor(amber)
        canvas.setLineWidth(1)
        canvas.line(0.45 * inch, 0.5 * inch, page_w - 0.45 * inch, 0.5 * inch)
        canvas.setFillColor(colors.HexColor("#5B6B80"))
        canvas.setFont("Helvetica", 7.5)
        canvas.drawString(0.45 * inch, 0.33 * inch, WATERMARK_TEXT)
        canvas.drawRightString(page_w - 0.45 * inch, 0.33 * inch, f"Page {doc.page}")
        canvas.restoreState()

    styles = getSampleStyleSheet()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=(page_w, page_h), leftMargin=0.45 * inch, rightMargin=0.45 * inch,
                            topMargin=0.85 * inch, bottomMargin=0.7 * inch, title=title, author="Shock Lab")
    grid = TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), navy),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("LINEBELOW", (0, 0), (-1, 0), 1.2, amber),
            ("FONTSIZE", (0, 0), (-1, -1), 7.5),
            ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F4F8")]),
            ("GRID", (0, 1), (-1, -1), 0.25, colors.HexColor("#D5DCE6")),
        ]
    )
    h3 = styles["Heading3"]
    h3.textColor = navy
    story = []
    kv = Table([[k, v] for k, v in settings + summary], hAlign="LEFT", colWidths=[2.4 * inch, 7.2 * inch])
    kv.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 9), ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                            ("TEXTCOLOR", (0, 0), (0, -1), navy)]))
    story += [kv, Spacer(1, 12)]
    for heading, df in tables:
        if df.empty:
            continue
        data = [list(map(str, df.columns))] + [[_fmt(v, c) for v, c in zip(row, df.columns)] for row in df.itertuples(index=False)]
        t = Table(data, hAlign="LEFT", repeatRows=1)
        t.setStyle(grid)
        story += [Paragraph(heading, h3), t, Spacer(1, 12)]
    story.append(Paragraph(disclaimer.replace("**", ""), styles["Italic"]))
    doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
    return buf.getvalue()
