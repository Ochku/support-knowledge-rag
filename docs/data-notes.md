# Data Notes: the corpus and its traps

Everything was fetched on **2026-10-30** from fossanalytics.com, with robots.txt checked for every URL and about 1 request per second. The full list is in `data/sources.tsv`: path, URL, kind, language, industries, product title, bytes and hash.

## Inventory (`data/raw/`, not in git)
| Folder | Files | What | Use |
|---|---|---|---|
| `products/en/` | 71 HTML | Every product in FOSS's product finder (all 7 industries) | **Core corpus** |
| `products/de/` | 8 HTML | German versions of 8 products (MilkoScan FT3, Infratec NOVA, MeatMaster II, WineScan 3, Kjeltec 9, Fossomatic 7, NIRS DS3 Feed, FoodScan 2 Meat) | Cross-lingual eval |
| `pdfs/` | 13 PDF (137 pages) | White papers, guides, an e-book: NIR calibration best practice, fibre methods, ANN + NIR, DSCC, fat analysis in meat, FT-IR standardisation, adulteration screening, Alphatec falling number… | **Core corpus**: method/"how-to" questions |
| `services/` | 11 HTML | Services, service agreements, training, digital services, SmartCare, report incident, certificates | Support-process questions |
| `company/` | 9 HTML | Who we are, mission, innovation, sustainability, digital, careers pages | Briefing, **not** the default QA corpus (see `decisions.md`) |
| `catalogue/products.json` | 1 JSON | 71 products with structured fields: `ProductTitle`, `WhatCanYouAnalyse`, `WhereCanYouAnalyse`, `ParametersInfo`, `AnalysisTime`, `Principle`, `Industries`, `Url` | Metadata, filters, lookup tool |

Industries (catalogue keys): `dairy` 20 · `feed-and-forage` 13 · `grain` 18 · `lab` 21 · `meat` 14 · `rmt` (raw milk testing) 9 · `wine` 4. Several products appear in more than one industry (e.g. Kjeltec in dairy, feed, lab and meat).

## Known traps (each one is a test)
1. **Quote and contact forms on every product and service page.** About 250 country names ("Afghanistan Albania Algeria…") and about 25 job titles ("Lab Manager Laborant…"), plus "Please provide your first name…". If you don't strip `<form>` and `<select>`, these become your most common chunks and BM25 will love them. → Test: no chunk contains `Afghanistan` or `Please provide`.
2. **The menu is repeated twice per page** (desktop and mobile nav) with ~150 link labels. Strip `<nav>`/`<header>`/`<footer>`; the real content starts after the breadcrumb ("Home › Products › …").
3. **Cookie and YouTube consent text** ("The content is hosted on YouTube.com (Third Party)…") appears in many sections. Strip it.
4. **Product pages have tabs** (Features · SmartCare · Technical Specification · Brochures and Papers · Value · Case Story). The content is in the HTML (not loaded by JS), so heading-aware chunking should keep these as sections.
5. **One PDF has broken font encoding:** `global-standard-adf_lignin-_1_.pdf` extracts as `/g72/g85/g3/…` glyph codes. Detect it with a letter-ratio check and exclude or flag it.
6. **Some PDFs have image-heavy first pages** (very little text on page 1, e.g. the NIR calibration guide). Don't judge a PDF by page 1.
7. **Trademark symbols:** `™` appears inside product names ("MilkoScan™ FT3"), and the catalogue `Teaser` has HTML entities (`&trade;`). Normalise them for matching ("milkoscan ft3"), or BM25 and exact-name lookups will miss.
8. **Duplicated text across products** (shared SmartCare blurbs, the same Kjeltec text in 4 industries). Expect near-duplicate chunks. Consider deduplicating by hash, or accept it and watch for diversity in the top-5.
9. **German pages share the English URL slug** (`/de-de/products/milkoscan-ft3`), so `lang` must come from the folder or manifest, not the slug.
10. **Not everything is answerable from the docs:** prices, delivery times, firmware and software bugs, and local service contacts are *not* in the corpus. Those make good unanswerable eval cases.

## Not included (and why)
- **35 brochures and datasheets** linked from product pages are served from `storage.pardot.com`, whose robots.txt disallows automated access. They're listed in `manual_downloads.md` for saving by hand in a browser (stretch S4).
- **FOSS IQX (iqx.net)** marketing pages: out of scope for a support-knowledge prototype.
- **Blog and knowledge articles:** good content, but not needed for the MVP. Add them later if the eval shows gaps in "how-to" questions.

## Licence and handling
FOSS's public content, fetched for personal, local evaluation only. **Don't commit `data/raw/` and don't publish it** (a public GitHub repo with this data would be redistribution). Code and the manifest are fine to publish.
