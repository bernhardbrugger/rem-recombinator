"""Embedding providers and cosine similarity.

Voyage ``voyage-3.5`` returns 1024-dimensional float vectors by default and
can emit 256, 512, 1024, or 2048 dimensions. The store records the width of
each stored vector. Do not mix providers or widths in one database.

``DeterministicEmbedder`` is a signed feature hash. It makes the pipeline
runnable offline. It is not a semantic model and must not be used for
scientific claims about novelty.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from typing import Protocol

import numpy as np

from rem_recombinator.config import ALLOWED_VOYAGE_DIMS, ExperimentConfig

logger = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    name: str

    def embed_text(self, text: str, input_type: str = "document") -> np.ndarray: ...

    def embed_batch(self, texts: list[str], input_type: str = "document") -> list[np.ndarray]: ...


def as_vector(value: np.ndarray) -> np.ndarray:
    """Return a contiguous finite float32 vector."""
    vector = np.ascontiguousarray(np.asarray(value, dtype=np.float32).reshape(-1))
    if vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError("embedding must be a non-empty finite vector")
    return vector


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in ``[-1, 1]``. A zero vector has similarity 0."""
    left = np.asarray(a, dtype=np.float64).reshape(-1)
    right = np.asarray(b, dtype=np.float64).reshape(-1)
    if left.shape != right.shape:
        raise ValueError(
            f"embedding dimension mismatch: {left.shape[0]} vs {right.shape[0]}"
        )
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom == 0.0:
        return 0.0
    return float(np.dot(left, right) / denom)


def unit_vector(vector: np.ndarray) -> np.ndarray:
    """L2-normalize. A zero vector is returned unchanged."""
    array = np.asarray(vector, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(array))
    if norm == 0.0:
        return array.astype(np.float32)
    return (array / norm).astype(np.float32)


class DeterministicEmbedder:
    """Signed feature hashing into a fixed dimension. Offline and deterministic."""

    name = "hashing"

    def __init__(self, dimensions: int = 32) -> None:
        if dimensions < 4:
            raise ValueError("hashing embedder needs at least 4 dimensions")
        self.dimensions = dimensions

    def embed_text(self, text: str, input_type: str = "document") -> np.ndarray:
        del input_type  # hashing has no query/document distinction
        vector = np.zeros(self.dimensions, dtype=np.float64)
        tokens = _TOKEN.findall(text.lower())
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        return unit_vector(vector)

    def embed_batch(self, texts: list[str], input_type: str = "document") -> list[np.ndarray]:
        return [self.embed_text(text, input_type=input_type) for text in texts]


class VoyageEmbedder:
    """Voyage AI client. Retries rate limits with backoff and logs to the module logger."""

    name = "voyage"

    def __init__(self, config: ExperimentConfig, max_retries: int = 5) -> None:
        if config.embedding_dimensions not in ALLOWED_VOYAGE_DIMS:
            raise ValueError(
                f"voyage output dimension {config.embedding_dimensions} is not supported"
            )
        self._config = config
        self._max_retries = max_retries
        self._client = None

    def _client_or_create(self):
        if self._client is None:
            import voyageai

            self._client = voyageai.Client(api_key=self._config.voyage_api_key)
        return self._client

    def _embed(self, texts: list[str], input_type: str):
        import voyageai

        client = self._client_or_create()
        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                return client.embed(
                    texts,
                    model=self._config.voyage_model,
                    input_type=input_type,
                    output_dimension=self._config.embedding_dimensions,
                )
            except voyageai.error.RateLimitError as exc:
                last_error = exc
                wait = 25 * (attempt + 1)
                logger.warning(
                    "Voyage rate limit, waiting %ss before retry %s/%s",
                    wait,
                    attempt + 1,
                    self._max_retries,
                )
                time.sleep(wait)
        raise RuntimeError(
            "Voyage remained rate-limited after retries. "
            "A payment method on the Voyage account raises the requests-per-minute cap."
        ) from last_error

    def embed_text(self, text: str, input_type: str = "document") -> np.ndarray:
        result = self._embed([text], input_type)
        return as_vector(result.embeddings[0])

    def embed_batch(self, texts: list[str], input_type: str = "document") -> list[np.ndarray]:
        if not texts:
            return []
        result = self._embed(texts, input_type)
        return [as_vector(vector) for vector in result.embeddings]


def build_embedder(name: str, config: ExperimentConfig) -> Embedder:
    if name == "hashing":
        return DeterministicEmbedder()
    if name == "voyage":
        config.validate_keys(need_anthropic=False, need_voyage=True)
        return VoyageEmbedder(config)
    raise ValueError(f"unknown embedder {name!r}; expected 'voyage' or 'hashing'")
