"""Vector store over 10-K chunks and the search_filings tool."""

import chromadb
from chromadb.utils import embedding_functions

from src.config import DATA_DIR, MODELS
from src.tools.filings import build_filing_chunks

_client = chromadb.PersistentClient(path=str(DATA_DIR / "chroma"))
_embed = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=MODELS["embeddings"])
_collection = _client.get_or_create_collection("filings", embedding_function=_embed)


def index_filings(tickers: list[str]) -> int:
    """Embed and store 10-K chunks. Safe to re-run (upsert by id)."""
    total = 0
    for t in tickers:
        chunks = build_filing_chunks(t)
        _collection.upsert(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[{k: c[k] for k in ("ticker", "section", "filed", "url")} for c in chunks],
        )
        total += len(chunks)
    return total


def search_filings(ticker: str, query: str, k: int = 3, section: str | None = None) -> list[dict]:
    """Top-k filing chunks for a ticker. Tool exposed to the earnings agent."""
    where = {"ticker": ticker}
    if section is not None:
        where = {"$and": [{"ticker": ticker}, {"section": section}]}
    res = _collection.query(query_texts=[query], n_results=k, where=where)
    return [
        {"text": doc, **meta, "distance": round(dist, 3)}
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]
