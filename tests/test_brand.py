import plotly.graph_objects as go

from shock_lab import brand
from shock_lab.report import to_pdf


def test_summary_card_escapes_user_text():
    html = brand.summary_card_html("<script>x</script>", 10_000, -1_000, -10.0, -1_200, -12.0, ("A&B", -500, -20.0), [])
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "A&amp;B" in html
    assert "-$1,000" in html and "lose" in html


def test_summary_card_gain_wording():
    html = brand.summary_card_html("Oil +20%", 10_000, 250, 2.5, 200, 2.0, None, [])
    assert "gain" in html and "+$250" in html


def test_style_chart_adds_watermark():
    fig = brand.style_chart(go.Figure(go.Bar(x=[1], y=[1])), 300)
    assert fig.layout.height == 300
    assert fig.layout.images[0].source.startswith("data:image/svg+xml;base64,")
    assert any(a.text == "SHOCK LAB" for a in fig.layout.annotations)


def test_step_links_match_anchors():
    for n in range(1, len(brand.STEPS) + 1):
        assert f'href="#step-{n}"' in brand.steps_html(3)
        assert f'id="step-{n}"' in brand.step_html(n, "Title")


def test_branded_pdf_builds():
    import pandas as pd

    pdf = to_pdf("Shock Lab report", [("Holdings", "1")], [], [("T", pd.DataFrame({"Impact ($)": [-1.0]}))], "x")
    assert pdf.startswith(b"%PDF")


def test_welcome_screen_images_exist():
    import re
    from pathlib import Path

    from shock_lab import landing

    html = landing.hero_shot_html() + landing.body_html()
    names = re.findall(r'src="app/static/([^"]+)"', html)
    assert len(names) == 5
    root = Path(__file__).resolve().parents[1] / "static"
    for n in names:
        assert (root / n).is_file(), n
