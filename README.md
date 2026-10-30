# support-knowledge-rag

A small, **measured** retrieval-augmented support assistant over public product documentation for food and agricultural analytics instruments (FOSS).

It uses Qdrant hybrid search (dense + BM25) for retrieval. Claude writes the answers with citations, or escalates to human support when the sources don't cover the question. An evaluation harness measures retrieval quality and answer quality, so changes can be compared against numbers rather than impressions.

> **Status:** work in progress. The environment, data manifest, design and evaluation spec are in place. The implementation is under way.

## Goals
- **Grounded answers:** every claim cites the document (and page) it came from.
- **Knows when not to answer:** questions the corpus can't answer, such as prices, error codes or local contacts, get escalated instead of guessed.
- **Multilingual:** a question in one language can be answered from sources in another, for example German questions over English documents.
- **Measured:** hit@k, recall@5 and MRR for retrieval, plus an LLM-judge rubric for answers, tracked per search mode and per question category.

## Docs
| Doc | What it covers |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | Ingestion, query and evaluation flows. Open [`docs/architecture.html`](docs/architecture.html) in a browser to see the rendered diagrams. |
| [`docs/data-notes.md`](docs/data-notes.md) | What's in the corpus and the known parsing traps |
| [`docs/eval-spec.md`](docs/eval-spec.md) | Metrics, eval case format and targets |
| [`docs/decisions.md`](docs/decisions.md) | Design decisions and open questions |
| [`docs/references.md`](docs/references.md) | Library docs and API gotchas |
| [`eval/rubric.md`](eval/rubric.md) | Rubric for the answer judge |

## Quickstart
Requirements: [uv](https://docs.astral.sh/uv/), Python 3.12, and Docker (optional).

```sh
uv sync                      # Python 3.12 venv + locked deps (dev group included)
cp .env.example .env         # add ANTHROPIC_API_KEY, or use `ant auth login`
docker compose up -d         # Qdrant at http://localhost:6333/dashboard
                             # no Docker? use embedded mode: leave QDRANT_URL empty, keep QDRANT_PATH
uv sync --group api          # optional: FastAPI service
uv sync --group mcp          # optional: MCP server
```

Embeddings run locally with [fastembed](https://github.com/qdrant/fastembed). Set `FASTEMBED_CACHE` (default `.cache/fastembed`) and pass it as `cache_dir` so the models aren't downloaded again on every run. Model choices are configured in `.env`.

## Repo layout
| Path | What |
|---|---|
| `src/` | Application code |
| `tests/` | Tests (chunking, metrics) |
| `eval/` | Example eval cases and the judge rubric. The full eval set lives in `eval/cases.jsonl`. |
| `docs/` | Design and evaluation docs |
| `data/sources.tsv` | Corpus manifest: source URL, kind, language, industries and hash for every document |
| `data/manual_downloads.md` | Documents that have to be downloaded by hand (robots.txt blocks automated fetching) |

## Data
The corpus consists of public pages and PDFs from FOSS's website. It is third-party content, so it is **not redistributed in this repository**. `data/raw/` and all derived artefacts (`data/processed/`, local Qdrant storage, `results/`) are gitignored. `data/sources.tsv` lists the source URL of every document so the corpus can be rebuilt.

This is an independent project. It is not affiliated with or endorsed by FOSS.
