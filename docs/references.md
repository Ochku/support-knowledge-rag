# References and Gotchas

## Locked versions (from `uv.lock`, 2026-10-30)
anthropic **2.3.0** · qdrant-client **1.20.0** (server image `qdrant/qdrant:v1.20.0`) · fastembed **0.9.1** · onnxruntime 1.31.1 · pypdf 6.21.4 · beautifulsoup4 4.15.0 · Python 3.12

## Claude API (anthropic SDK)
- Tool use overview: https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
- Structured outputs (for the judge): https://platform.claude.com/docs/en/build-with-claude/structured-outputs
- Citations (an alternative: pass passages as `document` blocks and let the API cite): https://platform.claude.com/docs/en/build-with-claude/citations
- Effort: https://platform.claude.com/docs/en/build-with-claude/effort
- Pricing (for the cost column): https://platform.claude.com/docs/en/about-claude/pricing
- Python SDK repo and migration notes: https://github.com/anthropics/anthropic-sdk-python (2.x is installed: check the README/MIGRATION for anything that changed from older examples you find online)

**Gotchas for `claude-opus-5-5` (from Anthropic's API reference):**
- **Thinking can't be disabled.** `thinking: {type: "disabled"}` returns a 400. Control depth with `output_config: {effort: ...}`. The **default effort is `medium`**: set it explicitly so runs are comparable.
- **Forced tool choice returns a 400.** `tool_choice: {type: "any"}` or `{type: "tool", ...}` is rejected. Use `auto` and say in the system prompt that it must search before answering.
- **Pass `response.content` back unchanged** in the tool loop, thinking blocks included. Don't rebuild the assistant message from text only.
- **Check `stop_reason` before reading `content`:** `tool_use` → run the tools · `end_turn` → done · `max_tokens` → truncated · `refusal` → declined. Return **all** tool results in **one** user message.
- **Structured output** uses `output_config: {format: {type: "json_schema", schema: ...}}` (or the SDK's `messages.parse` with a Pydantic model). The old `output_format` parameter is deprecated. Not combinable with document `citations`.
- **Refusal fallbacks:** Anthropic recommends opting into server-side fallbacks for Opus 5.5 (beta `server-side-fallback-2026-07-01`, `fallbacks: "default"`). Optional for this prototype. Just handle `stop_reason == "refusal"`.
- Prices (USD per 1M tokens, input/output): Opus 5.5 $4/$20 · Sonnet 5.5 $2/$10 · Haiku 5.5 $0.10/$0.50.

## Qdrant
- Documentation home: https://qdrant.tech/documentation/
- What to look up: **hybrid queries** (Query API, `prefetch` + `FusionQuery(fusion=RRF)`), **sparse vectors** (use `modifier=IDF` for BM25), **named vectors**, **payload indexes** (for the `kind`/`lang`/`industries` filters), **local mode** (`QdrantClient(path=...)`).
- Dashboard (Docker): http://localhost:6333/dashboard. Useful for inspecting collections and points.

## fastembed
- Repo and docs: https://github.com/qdrant/fastembed
- Models are cached in `.cache/fastembed/`: pass `cache_dir` or they'll be fetched again into `/tmp`.
- `TextEmbedding` has `embed`, `query_embed` and `passage_embed`. **Check whether it adds the e5 `query: `/`passage: ` prefixes for you.** If not, add them yourself (e5 quality drops without them). MiniLM doesn't need prefixes.
- `SparseTextEmbedding("Qdrant/bm25")` produces sparse vectors. Pair with `modifier=IDF` on the Qdrant sparse vector config.

## Key concepts (one sentence each)
- **Hybrid search / RRF:** run keyword (BM25) and semantic (dense) search separately and merge the rankings by reciprocal rank, so exact product names and paraphrases both work.
- **hit@k vs MRR:** "Was the right document in the top k?" vs "how high was it, on average?"
- **Groundedness vs correctness:** "Is every claim supported by what was retrieved?" vs "does it contain the right facts?" You can be correct but ungrounded (the model knew it anyway), and that's still a failure for support.
- **Judge calibration:** an LLM judge is a measuring instrument too. Check it against human labels before trusting it.
- **Why escalation matters:** a confident wrong answer costs a site visit or a customer's trust. "I don't know, here's who does" is a feature.

## FOSS-stack orientation (Sunday 12:45–14:00, reading only)
- Azure AI Foundry: projects, model deployments, evaluations; Azure AI Search vs Qdrant
- Boost.ai: how intent and flow-based conversation combines with generative answers and hand-over to agents or humans
