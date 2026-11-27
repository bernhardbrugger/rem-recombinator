"""
Thin wrapper around the Voyage API, so store.py/recombinator.py
aren't coupled directly to one provider.

Includes basic retry-with-backoff for rate limit errors — Voyage
accounts without a payment method on file are capped at 3 requests/min,
which a plain loop would blow through immediately.
"""
import time

import numpy as np
import voyageai

import config


_client = None
MAX_RETRIES = 5


def _get_client() -> voyageai.Client:
    global _client
    if _client is None:
        _client = voyageai.Client(api_key=config.VOYAGE_API_KEY)
    return _client


def _embed_with_retry(texts: list[str], input_type: str):
    for attempt in range(MAX_RETRIES):
        try:
            return _get_client().embed(
                texts, model=config.VOYAGE_MODEL, input_type=input_type
            )
        except voyageai.error.RateLimitError:
            wait = 25 * (attempt + 1)  # generous backoff for the 3 RPM free tier
            print(
                f"  Rate limit hit, waiting {wait}s before retry "
                f"({attempt + 1}/{MAX_RETRIES})..."
            )
            time.sleep(wait)
    raise RuntimeError(
        "Still rate-limited after several retries. Add a payment method at "
        "https://dashboard.voyageai.com/ to raise the limit (free tokens still apply)."
    )


def embed_text(text: str, input_type: str = "document") -> np.ndarray:
    """input_type: 'document' when storing, 'query' for search requests."""
    result = _embed_with_retry([text], input_type)
    return np.array(result.embeddings[0], dtype=np.float32)


def embed_batch(texts: list[str], input_type: str = "document") -> list[np.ndarray]:
    result = _embed_with_retry(texts, input_type)
    return [np.array(e, dtype=np.float32) for e in result.embeddings]
