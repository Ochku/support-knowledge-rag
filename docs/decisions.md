# Decision Log

Format: **ID · decision · why · status.** Add a line whenever you choose something. It records why each choice was made.

## Made (by PO, during setup)
| ID | Decision | Why | Status |
|---|---|---|---|
| D-001 | **Qdrant** as the vector store (Docker 1.20, or embedded local mode) | FOSS uses Qdrant. Supports named dense and sparse vectors and server-side hybrid fusion. | Decided |
| D-002 | **fastembed** for embeddings (ONNX, Hugging Face models, local) | Zero API cost and runs offline. Multilingual models. Same library family as Qdrant's examples. Models are pre-cached. | Decided |
| D-003 | **Hybrid retrieval** (dense + BM25, fused with RRF) as the default, with dense-only kept as the comparison | Product names and codes (*Kjeltec 8100*, *DSCC*) favour BM25; paraphrases and German favour dense | Decided; validated by the G1 experiment |
| D-004 | **No LangChain / LlamaIndex** | A thin pipeline is easier to explain and to measure. The posting lists LangChain, Hugging Face *or* OpenAI APIs; HF models (via fastembed) and a direct LLM SDK cover that. | Decided (revisit only if you want a talking point) |
| D-005 | **Corpus = public pages and papers only**, fetched politely, kept out of git | Legal and ethical, and reproducible via `data/sources.tsv` | Decided |
| D-006 | **Eval set written before tuning** | Avoids overfitting to your own questions | Decided |
| D-007 | Python 3.12, uv, `package = false` until code exists | ONNX runtime wheels are reliable on 3.12. The env works today without a package skeleton. | Decided |

## Open: your call (with a PO recommendation)
| ID | Question | Options | Recommendation | Decision |
|---|---|---|---|---|
| O-001 | Start embedding model | MiniLM (384d, fast) · e5-large (1024d, stronger, needs `query:`/`passage:` prefixes) | **Start with MiniLM** for fast iteration. If time allows, use e5-large as the experiment. | **`multilingual-e5-large`** (1024d) from the start. Changed from the recommendation: the cached MiniLM truncates input at 128 tokens, and 36% of English HTML sections and 73% of usable PDF pages are longer than that, so the dense vector would silently ignore the rest. e5-large takes 512 tokens. The corpus is only about 1,100 chunks, so speed is not the constraint. Needs the `query: ` / `passage: ` prefixes, which use a few tokens. MiniLM stays available as a comparison. |
| O-002 | Include `company/` pages in the QA index? | Yes · No · Yes, with `kind` filter | **No** in the default index (they answer "who is FOSS", not support questions). Keep them for the briefing. |  |
| O-003 | Catalogue use | Chunks · payload metadata · lookup tool · all three | **Metadata now, lookup tool as S1.** Don't also chunk it, or you'll double-count against the product pages. | **Payload metadata, passed through to the prompt.** Each product page chunk carries its catalogue fields (analysis time, principle, parameters, what and where you can analyse, industries), and they are shown to Claude next to the chunk text. Not chunked: duplicate catalogue chunks would fill top-k slots with the same fact. Payload is filterable but not searchable, so it only helps when the product's page chunk is retrieved or when it is printed in the prompt. Known gaps: a product missed by retrieval, comparisons, and complete lists (5 Kjeltec products, but top-5 chunks can hide some). Add a lookup tool (`name_contains` on `match_name`) only if the eval shows those failures. |
| O-004 | Chunk size | ~300 · ~500 tokens; overlap 0–15% | **~400 tokens, ~10% overlap.** Use the heading as the natural boundary first. | **~400 tokens, ~10% overlap, heading first**, as recommended. Measured in e5 tokens and kept under its 512 limit including the prepended title, heading and `passage: ` prefix. Merge tiny sections, split long ones at paragraph or sentence breaks, keep a table's rows with its header. |
| O-005 | Agent loop | Manual tool loop · SDK tool runner | **Manual loop.** You can explain every step, log every tool call and cost, and it maps to any provider (incl. Azure). |  |
| O-006 | Answer model and effort | Opus 5.5 (default, `medium` effort) · cheaper model | **Opus 5.5 at an explicit effort** for answers. Report cost per question. A cheaper model is a later experiment, not this weekend's. |  |
| O-007 | Judge model | Same as answer model · different model | **Same tier or stronger, never weaker.** Calibrate on 10 cases regardless. |  |
| O-008 | Qdrant mode | Docker · embedded | **Docker** if `docker compose up` works (you get the dashboard to show). Embedded mode is the fallback. |  |
| O-009 | CLI | argparse · nothing (scripts) | **argparse with subcommands** (`ingest`, `index`, `search`, `ask`, `eval`). No extra dependency. |  |
| O-010 | Experiment (G1) | hybrid vs dense · MiniLM vs e5 · chunk size | **hybrid vs dense** (cleanest story) |  |
