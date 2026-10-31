import pytest

from skrag.catalogue import CATALOGUE_JSON, load_catalogue, normalise_name

needs_data = pytest.mark.skipif(not CATALOGUE_JSON.exists(), reason="catalogue not present")


def test_normalise_name_drops_marks_and_case():
    assert normalise_name(" MilkoScan™  FT3 - Milk Analyser ") == "milkoscan ft3 - milk analyser"
    assert normalise_name("Kjeltec™ 9") == "kjeltec 9"


@needs_data
def test_catalogue_loads_all_products():
    products = load_catalogue()
    assert len(products) == 71
    assert len({p.id for p in products}) == 71
    ft3 = next(p for p in products if p.match_name.startswith("milkoscan ft3"))
    assert ft3.principle == "FTIR"
    assert ft3.analysis_time == "30 seconds"
    assert set(ft3.industries) == {"dairy", "rmt"}
    assert all(p.title == p.title.strip() and p.industries for p in products)
