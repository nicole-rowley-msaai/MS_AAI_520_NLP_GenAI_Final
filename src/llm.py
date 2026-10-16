"""Thin wrapper for structured LLM calls (Pydantic in, Pydantic out).

The provider is chosen from the model name: "claude-*" goes to Anthropic,
everything else to OpenAI. Every agent step goes through `structured`, so
swapping models is a config change, and tests can monkeypatch one function.
"""

import time

from pydantic import BaseModel, ValidationError

from src.config import MODELS, secret

MAX_TOKENS = 8192
RETRIES = 3

# If one provider keeps returning truncated or malformed JSON, the same
# call is made once on the other provider so a long run can finish.
FALLBACK = {"anthropic": MODELS["chain"], "openai": MODELS["writer"]}


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
    if resp.stop_reason != "end_turn":
        raise ValueError(f"{model} stopped early: {resp.stop_reason}")
    return resp.parsed_output


def _provider(model: str) -> str:
    return "anthropic" if model.startswith("claude") else "openai"


def structured(
    model: str, system: str, user: str, schema: type[BaseModel]
) -> BaseModel:
    """Call the model and parse its reply into `schema`.

    A reply that arrives truncated or malformed (a dropped stream, a
    max_tokens cut) is retried up to RETRIES times before the error
    surfaces, so one bad response does not end a multi-minute run.
    """
    try:
        return _with_retries(model, system, user, schema)
    except (ValidationError, ValueError) as e:
        other = FALLBACK[_provider(model)]
        if _provider(other) == _provider(model):
            raise
        why = str(e).splitlines()[0][:80]
        print(f"[llm] {model} failed {RETRIES}x ({why}); using {other}")
        return _with_retries(other, system, user, schema)


def _with_retries(model, system, user, schema):
    call = _anthropic if _provider(model) == "anthropic" else _openai
    for attempt in range(RETRIES):
        try:
            return call(model, system, user, schema)
        except (ValidationError, ValueError) as e:
            if attempt == RETRIES - 1:
                raise
            why = str(e).splitlines()[0][:80]
            name = schema.__name__
            print(f"[llm] retry {attempt + 1} for {name}: {why}")
            time.sleep(2 * (attempt + 1))
