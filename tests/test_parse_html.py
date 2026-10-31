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
<footer>Contact us</footer>
</body></html>
"""


def test_parse_html_keeps_content_and_drops_noise():
    title, sections = parse_html(HTML)
    text = "\n".join(s.text for s in sections)
    assert title == "AlphaTM 1"
    assert "Weight | 18 kg" in text
    assert "Real text" in text
    for noise in ("Menu", "Afghanistan", "Please provide", "Sign up", "YouTube", "Contact us"):
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
