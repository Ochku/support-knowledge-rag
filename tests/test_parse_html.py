import pytest

from skrag.parse_html import ROOT, parse_all, parse_html

needs_data = pytest.mark.skipif(not (ROOT / "data" / "raw").exists(), reason="data/raw not present")

HTML = """
<html><body>
<header><nav>Menu Products</nav></header>
<section class="panel header"><h1>Alpha<sup>TM</sup> 1</h1><h2>Tagline</h2></section>
<section class="panel table"><h2>Specs</h2>
  <table><tr><td>Weight</td><td>18 kg</td></tr></table></section>
<section class="panel newsletter"><h2>Sign up</h2>
  <form><select><option>Afghanistan</option></select>
  <p>Please provide your name</p></form></section>
<section class="panel"><p>Video: The content is hosted on YouTube.com (Third Party)</p>
  <p>Real text</p></section>
<section class="panel"><h2>Read the brochure: Better mastitis control</h2>
  <p>Get the insight you need</p></section>
<footer>Contact us</footer>
</body></html>
"""


def test_parse_html_keeps_content_and_drops_noise():
    title, sections = parse_html(HTML)
    text = "\n".join(s.text for s in sections)
    assert title == "AlphaTM 1"
    assert "Weight | 18 kg" in text
    assert "Real text" in text
    for noise in (
        "Menu",
        "Afghanistan",
        "Please provide",
        "Sign up",
        "YouTube",
        "Contact us",
        "brochure",
    ):
        assert noise not in text


@needs_data
def test_corpus_has_no_form_or_country_noise():
    pages = parse_all()
    assert len(pages) == 90
    for page in pages:
        for section in page.sections:
            assert "Afghanistan" not in section.text, page.path
            assert "Please provide your" not in section.text, page.path


@needs_data
def test_corpus_metadata_comes_from_manifest():
    pages = {p.path: p for p in parse_all()}
    de = pages["data/raw/products/de/milkoscan-ft3.html"]
    assert de.lang == "de"
    assert de.url.startswith("https://www.fossanalytics.com/de-de/")
    assert all(p.url and p.kind and p.lang for p in pages.values())
    assert all(p.sections for p in pages.values())
    assert all(p.product_title for p in pages.values() if p.kind == "products")


WINE_TABLE = """
<section class="panel"><h2>Technical specification</h2><table>
<tr><th>Components</th><th>Unit</th><th>Must</th><th>Must under fermentation</th>
  <th>Finished wine</th></tr>
<tr><td>Citric acid</td><td>g/L</td><td>0 - 5.4</td><td></td><td>0 - 1.0</td></tr>
<tr><td>Ammonia</td><td>mg/L</td><td></td><td>0 - 300</td><td></td></tr>
<tr><td>CO2</td><td>mg/L</td><td>0 - 2800</td><td></td><td></td></tr>
</table></section>
"""


def test_grid_table_with_gaps_tags_values_with_their_column():
    _, sections = parse_html(WINE_TABLE)
    text = sections[0].text
    assert "Citric acid (Unit: g/L; Must: 0 - 5.4; Finished wine: 0 - 1.0)" in text
    assert "Ammonia (Unit: mg/L; Must under fermentation: 0 - 300)" in text
    assert "Components | Unit | Must | Must under fermentation | Finished wine" in text


def test_tables_with_only_trailing_gaps_stay_plain_rows():
    """CO2 has empty cells only at the end, so no value can land under the wrong column."""
    html = """<section class="panel"><h2>Specs</h2><table>
    <tr><th>Components</th><th>Unit</th><th>Must</th><th>Finished wine</th></tr>
    <tr><td>CO2</td><td>mg/L</td><td>0 - 2800</td><td></td></tr>
    <tr><td>Ethanol</td><td>%</td><td>0 - 19</td><td>0 - 19</td></tr>
    </table></section>"""
    _, sections = parse_html(html)
    assert "CO2 | mg/L | 0 - 2800" in sections[0].text
    assert "(Unit" not in sections[0].text


@needs_data
def test_winescan_spec_keeps_stage_columns():
    page = next(p for p in parse_all() if p.path.endswith("products/en/winescan-3.html"))
    text = "\n".join(s.text for s in page.sections)
    assert "Citric acid (Unit: g/L; Must: 0 - 5.4; Finished wine: 0 - 1.0)" in text


MARKER_TABLE = """
<section class="panel"><h2>Technical specification</h2><table>
<tr><td colspan="4" style="background-color: #f1f1f1"><span>Included calibrations</span></td></tr>
<tr><td>Milk</td><td>Fat, protein</td></tr>
<tr><td><strong>Options</strong></td><td></td><td></td></tr>
<tr><td>Color module</td><td>Yes</td></tr>
<tr><td colspan="4">*Hammertec is specially designed for falling number, so it is only
  listed here as a footnote and not as a label for the rows above or below it.</td></tr>
</table></section>
"""


def test_marker_rows_become_headings_and_footnotes_do_not():
    _, sections = parse_html(MARKER_TABLE)
    lines = sections[0].text.splitlines()
    assert "#### Included calibrations" in lines
    assert "#### Options" in lines
    assert "Milk | Fat, protein" in lines
    assert not any(ln.startswith("####") and "Hammertec" in ln for ln in lines)


def test_marker_rows_inside_grid_tables_become_headings():
    html = WINE_TABLE.replace(
        "</table>",
        "<tr><td><strong>Options</strong></td><td></td><td></td><td></td><td></td></tr>"
        "<tr><td>Color module</td><td>Yes</td></tr></table>",
    )
    _, sections = parse_html(html)
    lines = sections[0].text.splitlines()
    assert "#### Options" in lines
    assert "Citric acid (Unit: g/L; Must: 0 - 5.4; Finished wine: 0 - 1.0)" in lines


def test_feature_matrix_with_blank_corner_keeps_models_aligned():
    html = """<section class="panel"><h2>Overview</h2><table>
    <tr><td></td><td>200</td><td>8100</td><td>8200</td></tr>
    <tr><td>Splash head</td><td>x</td><td>x</td><td>x</td></tr>
    <tr><td>Safety door</td><td></td><td>x</td><td>x</td></tr>
    <tr><td>Balance</td><td></td><td></td><td>x</td></tr>
    </table></section>"""
    _, sections = parse_html(html)
    lines = sections[0].text.splitlines()
    assert "200 | 8100 | 8200" in lines
    assert "Safety door (8100: x; 8200: x)" in lines
    assert "Balance (8200: x)" in lines
