"""Typed domain objects for fragments, recombinations, and critiques.

These are plain dataclasses rather than an ORM. SQLite remains the system of
record; the objects are the in-memory contract shared by the store, the
selector, the critic, and the exporters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Fragment:
    """One stored piece of knowledge and the vector used to compare it.

    ``domain`` is an explicit tag (for example ``physics`` or ``biology``),
    not something inferred silently at query time. ``metadata`` holds
    source-specific provenance such as an export path.
    """

    id: int | None
    text: str
    source: str
    domain: str
    created_at: float
    embedding: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding_dim: int = 0

    def __post_init__(self) -> None:
        if self.embedding is not None:
            self.embedding_dim = int(np.asarray(self.embedding).shape[0])


@dataclass(frozen=True)
class PairSelection:
    """A pair chosen by a recombination strategy, before any language model runs."""

    fragment_a: Fragment
    fragment_b: Fragment
    similarity: float
    strategy: str
    selection_score: float
    selection_notes: str


@dataclass(frozen=True)
class Hypothesis:
    """Structured recombination produced by the language model."""

    text: str
    rationale: str
    no_genuine_link: bool


@dataclass(frozen=True)
class NoveltyDecision:
    """Result of comparing a hypothesis vector with the stored corpus."""

    status: str
    novelty_score: float
    nearest_fragment_id: int | None
    nearest_similarity: float


@dataclass
class CritiqueReport:
    """Four-axis evaluation. ``accept`` is computed by the pipeline, not the model."""

    novelty: float
    logical_coherence: float
    falsifiability: float
    cross_domain_utility: float
    rationales: dict[str, str]
    improvement: str
    composite_score: float
    accept: bool
    parse_error: str | None = None

    def scores(self) -> dict[str, float]:
        return {
            "novelty": self.novelty,
            "logical_coherence": self.logical_coherence,
            "falsifiability": self.falsifiability,
            "cross_domain_utility": self.cross_domain_utility,
        }


@dataclass
class RecombinationCandidate:
    """A hypothesis plus the parents, strategy, and filter outcome that produced it."""

    id: int | None
    run_id: int | None
    parent_ids: list[int]
    strategy: str
    hypothesis: str
    rationale: str
    pair_similarity: float
    novelty_score: float | None
    nearest_fragment_id: int | None
    nearest_similarity: float | None
    filter_status: str
    selection_notes: str
    parent_snapshot: list[dict[str, Any]]
    critique: CritiqueReport | None = None
    materialized_fragment_id: int | None = None
    error: str | None = None
    created_at: float = 0.0


@dataclass
class RunResult:
    """One experiment execution and every candidate it persisted."""

    run_id: int
    hyperparameters: dict[str, Any]
    candidates: list[RecombinationCandidate]

    @property
    def accepted(self) -> list[RecombinationCandidate]:
        return [
            candidate
            for candidate in self.candidates
            if candidate.critique is not None and candidate.critique.accept
        ]
