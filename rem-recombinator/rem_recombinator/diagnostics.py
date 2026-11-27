"""Similarity distribution for threshold calibration."""

from __future__ import annotations

import random

import numpy as np

from rem_recombinator.embeddings import cosine_similarity
from rem_recombinator.models import Fragment


def similarity_sample(
    fragments: list[Fragment],
    *,
    samples: int,
    seed: int,
) -> np.ndarray:
    """Sample pair cosines. The sample is exhaustive when the corpus is small."""
    if len(fragments) < 2:
        return np.array([], dtype=np.float64)
    total = len(fragments) * (len(fragments) - 1) // 2
    rng = random.Random(seed)
    values: list[float] = []
    if total <= samples:
        for i in range(len(fragments)):
            for j in range(i + 1, len(fragments)):
                values.append(cosine_similarity(fragments[i].embedding, fragments[j].embedding))
    else:
        for _ in range(samples):
            left, right = rng.sample(fragments, 2)
            values.append(cosine_similarity(left.embedding, right.embedding))
    return np.array(values, dtype=np.float64)


def summarize(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {}
    return {
        "min": float(values.min()),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "max": float(values.max()),
        "n": float(values.size),
    }
