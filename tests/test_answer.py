from types import SimpleNamespace

import anthropic
import httpx2 as httpx
import pytest

from skrag.answer import (
    CHUNKS_PER_SEARCH,
    LIMIT_MESSAGE,
    MAX_SEARCHES,
    PRICES,
    SEARCH_TOOL,
    SYSTEM_PROMPT,
    Answerer,
    Passages,
    cited_numbers,
    cost_usd,
    page_label,
    sources_footer,
)
from skrag.search import Hit


def hit(id, url="https://example.com/a", text="body", heading="Spec", pages=None, **payload):
    start, end = pages or (None, None)
    return Hit(
        id=id, score=1.0, url=url, kind="products", lang="en", title="Title", heading=heading,
        page_start=start, page_end=end, text=text, payload=payload,
    )  # fmt: skip


# --- fakes ----------------------------------------------------------------------------------


def text_block(text):
    return SimpleNamespace(type="text", text=text)


def thinking_block():
    return SimpleNamespace(type="thinking", thinking="", signature="sig")


def tool_block(id, query="q", name="search_docs", **extra):
    return SimpleNamespace(type="tool_use", id=id, name=name, input={"query": query, **extra})


def response(content, stop_reason, in_t=100, out_t=10):
    usage = SimpleNamespace(
        input_tokens=in_t, output_tokens=out_t, cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )  # fmt: skip
    return SimpleNamespace(
        content=content, stop_reason=stop_reason, usage=usage, _request_id="req_1"
    )


class FakeClient:
    """Plays back scripted responses and keeps a copy of every request."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []
        self.messages = self

    def create(self, **kwargs):
        self.requests.append({**kwargs, "messages": [dict(m) for m in kwargs["messages"]]})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeSearcher:
    def __init__(self, hits_by_query=None):
        self.hits_by_query = hits_by_query or {}
        self.calls = []

    def search(self, query, mode, k, **filters):
        self.calls.append((query, mode, k, filters))
        return self.hits_by_query.get(query, [hit("c1")])


def answerer(client, searcher=None):
    return Answerer(
        searcher or FakeSearcher(), client=client, model="claude-opus-5-5", effort="medium"
    )


# --- passages and citations -----------------------------------------------------------------


def test_passages_are_numbered_once_across_searches():
    passages = Passages()
    assert passages.number(hit("a")) == 1
    assert passages.number(hit("b")) == 2
    assert passages.number(hit("a")) == 1
    assert sorted(passages.by_number) == [1, 2]


def test_rendered_passage_shows_number_source_pages_and_catalogue_fields():
    passages = Passages()
    chunk = hit("a", url="https://x/p.pdf", pages=(5, 6), analysis_time="30 seconds")
    text = passages.render([chunk])
    assert text.startswith("[1] Title › Spec (pp. 5-6)")
    assert "Source: https://x/p.pdf" in text
    assert "Analysis time: 30 seconds" in text
    assert text.endswith("body")


def test_page_label():
    assert page_label(hit("a")) == ""
    assert page_label(hit("a", pages=(3, 3))) == "p. 3"
    assert page_label(hit("a", pages=(3, 4))) == "pp. 3-4"


def test_no_results_say_so():
    assert "No passages found" in Passages().render([])


def test_cited_numbers_in_order_without_repeats():
    assert cited_numbers("A [2] and B [1][2], also [3, 4].") == [2, 1, 3, 4]
    assert cited_numbers("no citations, array[0] or [x]") == []


def test_sources_footer_lists_url_and_page_and_skips_unknown_numbers():
    passages = Passages()
    passages.number(hit("a", url="https://x/a"))
    passages.number(hit("b", url="https://x/b.pdf", pages=(7, 7)))
    assert sources_footer([2, 1, 9], passages) == "[2] https://x/b.pdf, p. 7\n[1] https://x/a"


def test_sources_footer_merges_chunks_of_the_same_page():
    passages = Passages()
    for chunk_id in ("a", "b", "c"):
        passages.number(hit(chunk_id, url="https://x/a"))
    passages.number(hit("d", url="https://x/p.pdf", pages=(2, 2)))
    passages.number(hit("e", url="https://x/p.pdf", pages=(3, 3)))
    assert sources_footer([3, 1, 4, 5], passages) == (
        "[3], [1] https://x/a\n[4] https://x/p.pdf, p. 2\n[5] https://x/p.pdf, p. 3"
    )


def test_cost_uses_the_model_prices_and_is_unknown_for_other_models():
    price = PRICES["claude-opus-5-5"]
    expected = (1000 * price.input + 500 * price.output) / 1_000_000
    assert cost_usd("claude-opus-5-5", 1000, 500, 0, 0) == pytest.approx(expected)
    assert cost_usd("some-future-model", 1000, 500, 0, 0) is None


# --- the loop -------------------------------------------------------------------------------


def test_an_answer_without_a_search_is_returned_as_is():
    client = FakeClient(response([text_block("Hello")], "end_turn"))
    result = answerer(client).answer("hi")
    assert result.text == "Hello"
    assert result.status == "ok"
    assert result.tool_calls == []
    assert result.requests == 1


def test_request_carries_system_prompt_tool_model_and_effort():
    client = FakeClient(response([text_block("x")], "end_turn"))
    answerer(client).answer("hi")
    request = client.requests[0]
    assert request["system"] == SYSTEM_PROMPT
    assert request["tools"] == [SEARCH_TOOL]
    assert request["model"] == "claude-opus-5-5"
    assert request["output_config"] == {"effort": "medium"}
    assert "tool_choice" not in request  # forcing a tool call is rejected by this model
    assert "thinking" not in request


def test_search_result_goes_back_with_the_assistant_turn_unchanged():
    assistant = [thinking_block(), tool_block("t1", "milkoscan time")]
    client = FakeClient(
        response(assistant, "tool_use"),
        response([text_block("30 seconds [1]")], "end_turn"),
    )
    searcher = FakeSearcher({"milkoscan time": [hit("c1", text="30 seconds")]})
    result = answerer(client, searcher).answer("How long?")
    second = client.requests[1]["messages"]
    assert second[1] == {"role": "assistant", "content": assistant}  # thinking block included
    assert second[2]["role"] == "user"
    block = second[2]["content"][0]
    assert block["type"] == "tool_result" and block["tool_use_id"] == "t1"
    assert "[1]" in block["content"] and "30 seconds" in block["content"]
    assert searcher.calls == [("milkoscan time", "hybrid", CHUNKS_PER_SEARCH, {})]
    assert result.text == "30 seconds [1]"
    assert result.cited == [1] and result.invalid_citations == []
    assert result.sources == "[1] https://example.com/a"
    assert result.retrieved_chunk_ids == ["c1"]


def test_parallel_searches_return_all_results_in_one_user_message():
    client = FakeClient(
        response([tool_block("t1", "a"), tool_block("t2", "b")], "tool_use"),
        response([text_block("done")], "end_turn"),
    )
    searcher = FakeSearcher({"a": [hit("c1")], "b": [hit("c2")]})
    answerer(client, searcher).answer("compare")
    results = client.requests[1]["messages"][2]["content"]
    assert [r["tool_use_id"] for r in results] == ["t1", "t2"]
    assert "[1]" in results[0]["content"] and "[2]" in results[1]["content"]


def test_filters_reach_the_search():
    client = FakeClient(
        response([tool_block("t1", "x", industry="dairy", lang="de")], "tool_use"),
        response([text_block("ok")], "end_turn"),
    )
    searcher = FakeSearcher()
    result = answerer(client, searcher).answer("q")
    assert searcher.calls[0][3] == {"industry": "dairy", "lang": "de"}
    assert result.tool_calls[0].filters == {"industry": "dairy", "lang": "de"}


def test_bad_tool_input_comes_back_as_an_error_result_without_searching():
    client = FakeClient(
        response(
            [tool_block("t1", "x", industry="cheese"), tool_block("t2", name="web_search")],
            "tool_use",
        ),
        response([text_block("ok")], "end_turn"),
    )
    searcher = FakeSearcher()
    result = answerer(client, searcher).answer("q")
    results = client.requests[1]["messages"][2]["content"]
    assert all(r.get("is_error") for r in results)
    assert searcher.calls == []
    assert [c.error != "" for c in result.tool_calls] == [True, True]


def test_the_search_cap_stops_further_searches_with_a_message():
    uses = [tool_block(f"t{i}", f"q{i}") for i in range(MAX_SEARCHES + 1)]
    client = FakeClient(response(uses, "tool_use"), response([text_block("ok")], "end_turn"))
    searcher = FakeSearcher()
    result = answerer(client, searcher).answer("q")
    results = client.requests[1]["messages"][2]["content"]
    assert len(searcher.calls) == MAX_SEARCHES
    assert results[-1]["content"] == LIMIT_MESSAGE and results[-1]["is_error"]
    assert len(result.tool_calls) == MAX_SEARCHES + 1


def test_a_citation_to_a_passage_never_retrieved_is_flagged():
    client = FakeClient(
        response([tool_block("t1")], "tool_use"),
        response([text_block("Claim [1] and another [7].")], "end_turn"),
    )
    result = answerer(client).answer("q")
    assert result.invalid_citations == [7]
    assert result.sources == "[1] https://example.com/a"


def test_usage_is_summed_over_every_request_and_priced():
    client = FakeClient(
        response([tool_block("t1")], "tool_use", in_t=1000, out_t=100),
        response([text_block("ok")], "end_turn", in_t=2000, out_t=300),
    )
    result = answerer(client).answer("q")
    assert (result.input_tokens, result.output_tokens, result.requests) == (3000, 400, 2)
    assert result.cost_usd == pytest.approx(cost_usd("claude-opus-5-5", 3000, 400, 0, 0))
    assert result.request_ids == ["req_1", "req_1"]
    assert result.latency_s > 0


@pytest.mark.parametrize("stop_reason", ["max_tokens", "refusal"])
def test_truncated_or_refused_answers_are_marked(stop_reason):
    client = FakeClient(response([text_block("partial")], stop_reason))
    result = answerer(client).answer("q")
    assert result.status == stop_reason


def test_a_loop_that_never_finishes_ends_with_a_round_limit():
    client = FakeClient(
        *[response([tool_block(f"t{i}")], "tool_use") for i in range(MAX_SEARCHES + 2)]
    )
    result = answerer(client).answer("q")
    assert result.status == "round_limit" and result.text == ""


def test_an_api_error_is_recorded_not_raised():
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    error = anthropic.APIConnectionError(request=request)
    result = answerer(FakeClient(error)).answer("q")
    assert result.status == "error" and "APIConnectionError" in result.error


def test_the_prompt_covers_grounding_escalation_and_language():
    for needle in ("[2]", "local FOSS", "language of the question", "do not guess"):
        assert needle in SYSTEM_PROMPT
