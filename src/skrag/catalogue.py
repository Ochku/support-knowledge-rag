"""Load the FOSS product catalogue as structured records.

How these records are used (chunks, payload metadata or a lookup tool) is decided in
``docs/decisions.md``; this module only parses and tidies them.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from pydantic import BaseModel

from skrag.parse_html import ROOT

CATALOGUE_JSON = ROOT / "data" / "raw" / "catalogue" / "products.json"

# Trademark marks inside product names ("MilkoScan™ FT3") break exact-name matching.
MARKS = re.compile("[™®℠]")


class Product(BaseModel):
    id: str
    title: str
    match_name: (
        str  # lowercase, marks removed, whitespace collapsed: "milkoscan ft3 - milk analyser"
    )
    url: str
    description: str
    analysis_time: str
    principle: str
    parameters: str
    what_you_can_analyse: str
    where_you_can_analyse: str
    industries: list[str]


def normalise_name(name: str) -> str:
    name = unicodedata.normalize("NFKC", MARKS.sub("", name))
    return re.sub(r"\s+", " ", name).strip().casefold()


def _tidy(value: str | None) -> str:
    return re.sub(r"\s+", " ", (value or "").replace("\xa0", " ")).strip()


def load_catalogue(path: Path = CATALOGUE_JSON) -> list[Product]:
    products = []
    for row in json.loads(path.read_text(encoding="utf-8")):
        title = _tidy(row["ProductTitle"])
        products.append(
            Product(
                id=row["Id"],
                title=title,
                match_name=normalise_name(title),
                url=_tidy(row["Url"]),
                description=_tidy(row["Body"]),
                analysis_time=_tidy(row["AnalysisTime"]),
                principle=_tidy(row["Principle"]),
                parameters=_tidy(row["ParametersInfo"]),
                what_you_can_analyse=_tidy(row["WhatCanYouAnalyse"]),
                where_you_can_analyse=_tidy(row["WhereCanYouAnalyse"]),
                industries=list(row["Industries"]),
            )
        )
    return products
