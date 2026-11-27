"""Compatibility shim. The implementation is :mod:`rem_recombinator.embeddings`."""

from rem_recombinator.config import load_config
from rem_recombinator.embeddings import (
    DeterministicEmbedder,
    VoyageEmbedder,
    build_embedder,
    cosine_similarity,
)


def embed_text(text: str, input_type: str = "document"):
    """Embed one string with Voyage, using the environment configuration."""
    config = load_config()
    config.validate_keys(need_anthropic=False, need_voyage=True)
    return VoyageEmbedder(config).embed_text(text, input_type=input_type)


def embed_batch(texts: list[str], input_type: str = "document"):
    config = load_config()
    config.validate_keys(need_anthropic=False, need_voyage=True)
    return VoyageEmbedder(config).embed_batch(texts, input_type=input_type)


__all__ = [
    "DeterministicEmbedder",
    "VoyageEmbedder",
    "build_embedder",
    "cosine_similarity",
    "embed_batch",
    "embed_text",
]
