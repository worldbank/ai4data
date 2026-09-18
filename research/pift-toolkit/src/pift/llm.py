"""
Provider-agnostic LLM client for synthetic query generation.

Two real providers are supported out of the box, selected by name:
  - ``anthropic``: uses the ``anthropic`` SDK and ``ANTHROPIC_API_KEY``.
  - ``openai``: uses the ``openai`` SDK and ``OPENAI_API_KEY``.

A third provider, ``heuristic``, needs no API key and no network. It derives
simple queries directly from the record fields. It exists so the example runs
end to end offline and so tests do not depend on a paid API. It is not a
substitute for a real model when you train for production.

Add another provider by implementing ``complete(system, user) -> str`` and
registering it in :func:`get_client`.
"""

from __future__ import annotations

import os


class AnthropicClient:
    def __init__(self, model: str):
        import anthropic

        self.model = model
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def complete(self, system: str, user: str) -> str:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


class OpenAIClient:
    def __init__(self, model: str):
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def complete(self, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""


class HeuristicClient:
    """No-API fallback. Ignores the prompt and is driven by generate.py instead.

    The generator special-cases this client so it can build queries directly
    from fields; ``complete`` is therefore never called in normal use.
    """

    def __init__(self, model: str = "heuristic"):
        self.model = model

    def complete(self, system: str, user: str) -> str:  # pragma: no cover
        raise RuntimeError("HeuristicClient.complete should not be called")


def get_client(provider: str, model: str):
    provider = provider.lower()
    if provider == "anthropic":
        return AnthropicClient(model)
    if provider == "openai":
        return OpenAIClient(model)
    if provider in ("heuristic", "none", "offline"):
        return HeuristicClient(model)
    raise ValueError(f"unknown LLM provider: {provider!r}")
