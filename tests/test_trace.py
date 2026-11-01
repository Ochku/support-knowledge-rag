from types import SimpleNamespace

from skrag.answer import Answerer
from skrag.search import Hit
from skrag.trace import TracingClient, TracingSearcher


def text_block(text):
    return SimpleNamespace(type="text", text=text)


def tool_block(id, query):
    return SimpleNamespace(type="tool_use", id=id, name="search_docs", input={"query": query})


def response(content, stop_reason):
    usage = SimpleNamespace(input_tokens=10, output_tokens=5)
    return SimpleNamespace(content=content, stop_reason=stop_reason, usage=usage)


class FakeClient:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.messages = self

    def create(self, **kwargs):
        return self.responses.pop(0)


class FakeSearcher:
    def search(self, query, mode, k, **filters):
        return [
            Hit(
                id="c1",
                score=0.5,
                url="https://example.com/a",
                kind="products",
                lang="en",
                title="Title",
                heading="Spec",
                text="30 seconds",
                payload={},
            )  # fmt: skip
        ]


def run(*responses, traced=True):
    lines = []
    searcher, client = FakeSearcher(), FakeClient(*responses)
    if traced:
        searcher, client = (
            TracingSearcher(searcher, lines.append),
            TracingClient(client, lines.append),
        )
    result = Answerer(searcher, client=client, model="claude-opus-5-5", effort="medium").answer(
        "q?"
    )
    return result, "\n".join(lines)


def script():
    return (
        response([tool_block("t1", "milkoscan")], "tool_use"),
        response([text_block("30 seconds [1]")], "end_turn"),
    )


def test_the_trace_shows_instructions_once_and_each_step():
    result, output = run(*script())
    assert output.count("--- system (the instructions) ---") == 1
    assert "REQUEST 1" in output and "REQUEST 2" in output
    assert "stop_reason 'tool_use'" in output and "stop_reason 'end_turn'" in output
    assert "[user] q?" in output
    assert "our code runs the search: 'milkoscan'" in output
    assert "tool_result for t1" in output
    assert result.text == "30 seconds [1]"


def test_the_second_request_prints_only_the_new_messages():
    _, output = run(*script())
    second = output.split("REQUEST 2")[1]
    assert "[user] q?" not in second  # already shown with the first request
    assert "tool_result for t1" in second


def test_the_trace_changes_nothing_about_the_result():
    plain, _ = run(*script(), traced=False)
    traced, _ = run(*script())
    assert (traced.text, traced.status, traced.requests) == (
        plain.text,
        plain.status,
        plain.requests,
    )
