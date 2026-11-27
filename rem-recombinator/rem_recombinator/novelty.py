"""Novelty filter.

A hypothesis is discarded when its embedding is too close to anything already
stored (paraphrase) or too far from everything stored (no anchor in the corpus).

    N(h) = 1 - max_f cos(e_h, e_f)

Reject when max cos > novelty_ceiling (default 0.88) or max cos < incoherence_floor
(default 0.05). Parents are included in the corpus: a restatement of a parent
is not novel.
"""

from __future__ import annotations

import numpy as np

from rem_recombinator.embeddings import cosine_similarity
from rem_recombinator.models import Fragment, NoveltyDecision


def status_from_nearest(nearest: float, *, ceiling: float, floor: float) -> str:
    """Same rule the filter uses, so a threshold sweep cannot drift from it."""
    if nearest > ceiling:
        return "too_similar"
    if nearest < floor:
        return "incoherent"
    return "kept"


def assess_novelty(
    hypothesis: np.ndarray,
    fragments: list[Fragment],
    *,
    ceiling: float,
    floor: float,
) -> NoveltyDecision:
    """Compare ``hypothesis`` with every stored fragment."""
    if not fragments:
        return NoveltyDecision(
            status="incoherent",
            novelty_score=1.0,
            nearest_fragment_id=None,
            nearest_similarity=0.0,
        )
    nearest_id: int | None = None
    nearest = -1.0
    for fragment in fragments:
        similarity = cosine_similarity(hypothesis, fragment.embedding)
        if similarity > nearest:
            nearest = similarity
            nearest_id = fragment.id
    status = status_from_nearest(nearest, ceiling=ceiling, floor=floor)
    return NoveltyDecision(
        status=status,
        novelty_score=1.0 - max(nearest, 0.0),
        nearest_fragment_id=nearest_id,
        nearest_similarity=nearest,
    )
