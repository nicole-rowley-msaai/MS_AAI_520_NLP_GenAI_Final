"""Thin wrapper for structured LLM calls (Pydantic in, Pydantic out).

The provider is chosen from the model name: "claude-*" goes to Anthropic,
everything else to OpenAI. Every agent step goes through `structured`, so
swapping models is a config change, and tests can monkeypatch one function.
"""

from pydantic import BaseModel

from src.config import secret

MAX_TOKENS = 4096


def _openai(model: str, system: str, user: str, schema: type[BaseModel]):
    from openai import OpenAI

    client = OpenAI(api_key=secret("OPENAI_API_KEY"))
    resp = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        text_format=schema,
    )
    return resp.output_parsed


def _anthropic(model: str, system: str, user: str, schema: type[BaseModel]):
    from anthropic import Anthropic

    client = Anthropic(api_key=secret("ANTHROPIC_API_KEY"))
    resp = client.messages.parse(
        model=model,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": user}],
        output_format=schema,
    )
    return resp.parsed_output


def structured(model: str, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
    """Call the model and parse its reply into `schema`."""
    if model.startswith("claude"):
        return _anthropic(model, system, user, schema)
    return _openai(model, system, user, schema)
