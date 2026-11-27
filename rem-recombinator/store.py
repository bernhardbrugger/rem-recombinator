"""Compatibility shim. The implementation is :mod:`rem_recombinator.store`."""

from rem_recombinator.embeddings import cosine_similarity
from rem_recombinator.models import Fragment
from rem_recombinator.store import Store

__all__ = ["Fragment", "Store", "cosine_similarity"]
