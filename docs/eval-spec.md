# Evaluation Spec

**Rule #1: write `eval/cases.jsonl` before you tune anything.** Otherwise you're grading your own homework.

## Case format (one JSON object per line)
| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | str | yes | e.g. `spec-001`, `xling-003`, `unans-002` |
| `question` | str | yes | Phrase it the way a support engineer or customer would. Not copy-pasted from the page. |
| `lang` | str | yes | `en`, `de`, `nl`, `da`… The language the **answer** must be in |
| `category` | str | yes | See the mix below |
| `answerable` | bool | yes | `false` → the correct behaviour is to escalate |
| `expected_sources` | list[str] | if answerable | Source URLs (as in `data/sources.tsv`). Any one counts as a hit. |
| `answer_points` | list[str] | if answerable | Short facts the answer must contain, e.g. `"30 seconds"` |
| `filters` | object | no | e.g. `{"industries": "dairy"}` if the question names an industry |
| `notes` | str | no | Why this case is interesting |

See `eval/cases.example.jsonl` for 5 worked examples, checked against the raw docs.

## Category mix (35 cases)
| Category | # | What it tests | Example |
|---|---|---|---|
| `spec` | 10 | Single fact from one product page | "How long does a MilkoScan FT3 analysis take?" |
| `method` | 5 | How-to and method knowledge from the white papers | "What's best practice for building a global NIR calibration?" |
| `which-product` | 5 | Mapping a need to a product (retrieval across many pages) | "Which instrument detects bone fragments in raw meat in-line?" |
| `multi-hop` | 3 | Needs 2 sources or a comparison | "What's the difference between the Kjeltec 9 Analyser and the Kjeltec 8100?" |
| `xling` | 5 | Question in DE (or NL/DA), answer from EN or DE pages | "Wie lange dauert eine Analyse mit dem MilkoScan FT3?" |
| `unans` | 5 | Not in the docs: price, delivery time, a specific error code, a local contact, a competitor | "What does an Infratec NOVA cost?" |
| `tricky` | 2 | Ambiguous or misleading premise | A product name that doesn't exist, or a mix-up between two products |

## Retrieval metrics (per search mode: dense, sparse, hybrid)
- **hit@k** (k = 1, 3, 5): share of answerable cases where any expected source appears in the top-k
- **recall@5:** share of the expected sources found in the top-5 (matters for multi-hop)
- **MRR:** mean of 1/rank of the first relevant result (0 if none in the top-k)
- Count a hit at the **document URL** level, not the chunk level: you label sources, not chunks.
- Report overall **and per category.** Cross-lingual and which-product cases are where hybrid vs dense differ.

## Answer metrics
| Metric | How | Target |
|---|---|---|
| **Correctness** | Judge: all `answer_points` present and not contradicted | ≥ 80% of answerable cases |
| **Groundedness** | Judge: every claim supported by the cited passages | ≥ 90% |
| **Citation validity** | Deterministic: every cited `[n]` refers to a passage actually retrieved in this run | 100% |
| **Refusal accuracy** | Deterministic + judge: `answerable=false` → escalates, no invented facts | 5/5 |
| **Over-refusal** | Answerable cases where it wrongly escalated | report the count |
| **Language match** | Deterministic (a language-ID heuristic or the judge) | 100% |
| **Cost** | Σ tokens × price, per question: mean and p95 | report |
| **Latency** | Wall clock per question: mean and p95 | report |

## Judge
- Rubric in `eval/rubric.md`. Return **structured output**: per criterion `pass: bool` plus `reason: str`.
- Give the judge the question, `answer_points`, the retrieved passages and the answer. **Not** your expected sources: it should judge grounding against what was actually retrieved.
- **Calibrate:** hand-label 10 answers before you trust it, and report the agreement (e.g. "judge agreed with me on 9/10").
- Judge with the same model or a stronger one, never a weaker one. Note the model in the results.

## Results file (`results/<timestamp>-<config>.json`)
Store: the config (embedding model, search mode, k, chunk size, answer model, effort, prompt version), aggregate metrics, and per-case records (retrieved IDs and URLs, answer, verdicts, tokens, cost, latency). The README table is generated from these files and never typed by hand.

## The one experiment
Change **one** variable. Recommended: **dense-only vs hybrid** at the same k and embedding model, because it's cheap and the story is clean ("BM25 helps on exact product names like *Kjeltec 8100*; dense helps on paraphrases and German"). The alternative is MiniLM vs e5-large.
