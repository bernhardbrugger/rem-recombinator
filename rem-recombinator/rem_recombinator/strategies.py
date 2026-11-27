"""Deterministic pair selection.

Three strategies share one implementation pattern: enumerate eligible pairs,
score them, and return the top ``n`` in descending score. Ties break by
fragment id, so a fixed corpus and config produce the same pairs.

Semantic bridge
    Greedy leader clustering (leaders chosen by ascending id, link threshold
    ``cluster_link_threshold``) partitions the corpus. A pair is eligible when
    the members sit in different clusters and their cosine lies in
    ``[min_similarity, max_similarity]``. The score is the novelty of the
    L2-normalized midpoint against every other fragment. That is a geometric
    stand-in for a concept that would sit between two distant regions, not a
    claim that the clustering is optimal.

Analogical transfer
    Eligible pairs have different domain tags and a cosine inside
    ``[analogical_min, analogical_max]``. The score is the cosine itself:
    within the band, a stronger shared direction is a better analogy. Fragment
    A is the lexicographically smaller domain.

Dialectic synthesis
    Eligible pairs carry at least one lexical opposition and a cosine inside
    the dialectic band. The score is ``tension * (1 - |cos - target|)``.
    Opposition is a documented cue list, not a semantic entailment check.
"""

from __future__ import annotations

import random
import re
from collections.abc import Iterator

import numpy as np

from rem_recombinator.config import STRATEGIES, ExperimentConfig
from rem_recombinator.embeddings import cosine_similarity, unit_vector
from rem_recombinator.models import Fragment, PairSelection

OPPOSITION_PAIRS: tuple[tuple[str, str], ...] = (
    ("always", "never"),
    ("increase", "decrease"),
    ("increases", "decreases"),
    ("increased", "decreased"),
    ("confirm", "disconfirm"),
    ("support", "refute"),
    ("cause", "prevent"),
    ("necessary", "contingent"),
    ("stable", "unstable"),
    ("positive", "negative"),
    ("true", "false"),
    ("central", "peripheral"),
    ("invariant", "context-dependent"),
)


def opposition_tension(text_a: str, text_b: str) -> int:
    """Count opposition cues with one pole in each text."""
    left = text_a.lower()
    right = text_b.lower()
    tension = 0
    for first, second in OPPOSITION_PAIRS:
        if (_contains(left, first) and _contains(right, second)) or (
            _contains(left, second) and _contains(right, first)
        ):
            tension += 1
    return tension


def assign_clusters(fragments: list[Fragment], link_threshold: float) -> dict[int, int]:
    """Map each fragment id to a leader id.

    A fragment becomes a new leader when its cosine to every existing leader
    is strictly below ``link_threshold``. Otherwise it joins the nearest leader.
    """
    ordered = sorted(fragments, key=lambda fragment: _id(fragment))
    leaders: list[Fragment] = []
    assignment: dict[int, int] = {}
    for fragment in ordered:
        best_similarity = -1.0
        best_leader: Fragment | None = None
        for leader in leaders:
            similarity = cosine_similarity(fragment.embedding, leader.embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_leader = leader
        if best_leader is None or best_similarity < link_threshold:
            leaders.append(fragment)
            assignment[_id(fragment)] = _id(fragment)
        else:
            assignment[_id(fragment)] = _id(best_leader)
    return assignment


def midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Equal-weight interpolation. Each parent is normalized before the sum."""
    return unit_vector(unit_vector(a).astype(np.float64) + unit_vector(b).astype(np.float64))


def select_pairs(
    fragments: list[Fragment],
    strategy: str,
    config: ExperimentConfig,
    rng: random.Random,
    n: int,
) -> list[PairSelection]:
    """Return up to ``n`` pairs for one strategy."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}")
    if n < 1 or len(fragments) < 2:
        return []
    clusters = assign_clusters(fragments, config.cluster_link_threshold)
    ranked: list[PairSelection] = []
    for left, right in _iter_pairs(fragments, rng, config.pair_enumeration_cap):
        selection = _score_pair(left, right, strategy, config, fragments, clusters)
        if selection is not None:
            ranked.append(selection)
    ranked.sort(key=lambda item: (-item.selection_score, _id(item.fragment_a), _id(item.fragment_b)))
    return ranked[:n]


def _score_pair(
    left: Fragment,
    right: Fragment,
    strategy: str,
    config: ExperimentConfig,
    fragments: list[Fragment],
    clusters: dict[int, int],
) -> PairSelection | None:
    similarity = cosine_similarity(left.embedding, right.embedding)
    if strategy == "semantic_bridge":
        return _bridge(left, right, similarity, config, fragments, clusters)
    if strategy == "analogical_transfer":
        return _analogy(left, right, similarity, config)
    return _dialectic(left, right, similarity, config)


def _bridge(
    left: Fragment,
    right: Fragment,
    similarity: float,
    config: ExperimentConfig,
    fragments: list[Fragment],
    clusters: dict[int, int],
) -> PairSelection | None:
    if clusters[_id(left)] == clusters[_id(right)]:
        return None
    if not config.min_similarity_for_candidate <= similarity <= config.max_similarity_for_candidate:
        return None
    ordered = _by_id(left, right)
    point = midpoint(ordered[0].embedding, ordered[1].embedding)
    if float(np.linalg.norm(point)) == 0.0:
        return None
    exclude = {_id(ordered[0]), _id(ordered[1])}
    nearest = 0.0
    for fragment in fragments:
        if _id(fragment) in exclude:
            continue
        nearest = max(nearest, cosine_similarity(point, fragment.embedding))
    score = 1.0 - nearest
    notes = (
        f"cross-cluster pair cosine {similarity:.4f} inside "
        f"[{config.min_similarity_for_candidate:.2f}, {config.max_similarity_for_candidate:.2f}]; "
        f"midpoint max cosine to other fragments {nearest:.4f}"
    )
    return PairSelection(ordered[0], ordered[1], similarity, "semantic_bridge", score, notes)


def _analogy(
    left: Fragment,
    right: Fragment,
    similarity: float,
    config: ExperimentConfig,
) -> PairSelection | None:
    domain_a = _domain(left)
    domain_b = _domain(right)
    if domain_a == domain_b or domain_a == "unspecified" or domain_b == "unspecified":
        return None
    if not config.analogical_min_similarity <= similarity <= config.analogical_max_similarity:
        return None
    source, target = (left, right) if domain_a <= domain_b else (right, left)
    notes = (
        f"domains {_domain(source)} -> {_domain(target)}; "
        f"pair cosine {similarity:.4f} inside "
        f"[{config.analogical_min_similarity:.2f}, {config.analogical_max_similarity:.2f}]"
    )
    return PairSelection(source, target, similarity, "analogical_transfer", similarity, notes)


def _dialectic(
    left: Fragment,
    right: Fragment,
    similarity: float,
    config: ExperimentConfig,
) -> PairSelection | None:
    if not config.dialectic_min_similarity <= similarity <= config.dialectic_max_similarity:
        return None
    tension = opposition_tension(left.text, right.text)
    if tension < 1:
        return None
    ordered = _by_id(left, right)
    distance = abs(similarity - config.dialectic_target_similarity)
    score = tension * (1.0 - distance)
    notes = (
        f"lexical tension {tension}; pair cosine {similarity:.4f}; "
        f"target {config.dialectic_target_similarity:.2f}"
    )
    return PairSelection(ordered[0], ordered[1], similarity, "dialectic_synthesis", score, notes)


def _iter_pairs(
    fragments: list[Fragment],
    rng: random.Random,
    cap: int,
) -> Iterator[tuple[Fragment, Fragment]]:
    ordered = sorted(fragments, key=_id)
    total = len(ordered) * (len(ordered) - 1) // 2
    if total <= cap:
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                yield ordered[i], ordered[j]
        return
    seen: set[tuple[int, int]] = set()
    attempts = 0
    limit = cap * 5
    while len(seen) < cap and attempts < limit:
        attempts += 1
        i, j = sorted(rng.sample(range(len(ordered)), 2))
        key = (_id(ordered[i]), _id(ordered[j]))
        if key in seen:
            continue
        seen.add(key)
        yield ordered[i], ordered[j]


def _contains(text: str, term: str) -> bool:
    return re.search(rf"\b{re.escape(term)}\b", text) is not None


def _domain(fragment: Fragment) -> str:
    return (fragment.domain or "unspecified").strip().lower() or "unspecified"


def _id(fragment: Fragment) -> int:
    if fragment.id is None:
        raise ValueError("selection requires persisted fragments with ids")
    return fragment.id


def _by_id(left: Fragment, right: Fragment) -> tuple[Fragment, Fragment]:
    if _id(left) <= _id(right):
        return left, right
    return right, left
