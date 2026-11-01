"""Print every step of an answer run, to see how the loop works.

``TracingClient`` wraps the Anthropic client and prints each request (the full instructions and
tool definition once, then only the messages added since the last request) and each response.
``TracingSearcher`` wraps the searcher and prints each search with the chunks it returned.
Neither changes what is sent or returned.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

RULE = "=" * 78


class TracingClient:
    def __init__(self, client: Any, out: Callable[[str], None] = print) -> None:
        self.client = client
        self.messages = self  # so that client.messages.create(...) reaches create() below
        self.out = out
        self.requests = 0
        self._shown = 0  # how many messages were already printed

    def create(self, **kwargs: Any) -> Any:
        self.requests += 1
        self.out(f"\n{RULE}\nREQUEST {self.requests}: sent to Claude\n{RULE}")
        if self.requests == 1:
            self.out(f"model {kwargs['model']}, max_tokens {kwargs['max_tokens']}")
            self.out(f"output_config {kwargs.get('output_config')}")
            self.out(f"\n--- system (the instructions) ---\n{kwargs['system']}")
            self.out(
                f"\n--- tools (what Claude may call) ---\n{json.dumps(kwargs['tools'], indent=2)}"
            )
        sent = kwargs["messages"]
        total = len(sent)
        self.out(f"\n--- messages: {total} in the conversation, {total - self._shown} new ---")
        for message in sent[self._shown :]:
            self.out(_describe(message))
        self._shown = total + 1  # the reply to this request is added after it
        response = self.client.messages.create(**kwargs)
        self.out(f"\n{RULE}\nRESPONSE {self.requests}: from Claude\n{RULE}")
        usage = response.usage
        self.out(
            f"stop_reason {response.stop_reason!r}, "
            f"{usage.input_tokens} tokens in, {usage.output_tokens} tokens out"
        )
        for block in response.content:
            self.out(_describe_block(block))
        return response


class TracingSearcher:
    def __init__(self, searcher: Any, out: Callable[[str], None] = print) -> None:
        self.searcher = searcher
        self.out = out

    def search(self, query: str, mode: str, k: int, **filters: str) -> list[Any]:
        hits = self.searcher.search(query, mode, k, **filters)
        self.out(f"\n>>> our code runs the search: {query!r}, filters {filters}, {mode}, k={k}")
        for rank, hit in enumerate(hits, start=1):
            self.out(f"    {rank}. score {hit.score:.3f}  {hit.title} › {hit.heading[:50]}")
        return hits


def _describe(message: dict[str, Any]) -> str:
    role, content = message["role"], message["content"]
    if isinstance(content, str):
        return f"[{role}] {content}"
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_result":
            error = " (error)" if block.get("is_error") else ""
            parts.append(f"  tool_result for {block['tool_use_id']}{error}:\n{block['content']}")
        else:
            parts.append("  " + _describe_block(block))
    return f"[{role}]\n" + "\n".join(parts)


def _describe_block(block: Any) -> str:
    kind = getattr(block, "type", None)
    if kind == "text":
        return f"text: {block.text}"
    if kind == "tool_use":
        return f"tool_use {block.name} (id {block.id}): {block.input}"
    if kind == "thinking":
        return f"thinking: hidden by default ({len(block.thinking)} characters shown)"
    return f"{kind} block"
