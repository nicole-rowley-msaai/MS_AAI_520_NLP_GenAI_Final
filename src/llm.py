"""Thin wrapper for structured LLM calls (Pydantic in, Pydantic out)."""

from openai import OpenAI
from pydantic import BaseModel

from src.config import OPENAI_API_KEY

_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


def structured(model: str, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
    """Call the model and parse its reply into `schema`."""
    if _client is None:
        raise RuntimeError("OPENAI_API_KEY is not set in .env")
    resp = _client.responses.parse(
        model=model,
        input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        text_format=schema,
    )
    return resp.output_parsed
