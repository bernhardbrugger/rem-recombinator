"""Selection strategies on a hand-built corpus."""

import random

import pytest

from rem_recombinator.embeddings import cosine_similarity
from rem_recombinator.strategies import assign_clusters, opposition_tension, select_pairs
from tests.fakes import unit


def _seed(store, text, domain, vector):
    store.add_fragment(text, vector, source="test", domain=domain)


def test_opposition_uses_word_boundaries():
    assert opposition_tension("It always increases.", "It never increases.") == 1
    assert opposition_tension("instrument", "false friend") == 0


def test_clusters_split_distant_vectors(store):
    _seed(store, "same-a", "physics", unit(1, 0, 0))
    _seed(store, "same-b", "physics", unit(1, 0, 0))
    _seed(store, "far", "biology", unit(0, 1, 0))
    fragments = store.all_fragments()
    clusters = assign_clusters(fragments, link_threshold=0.55)
    assert clusters[fragments[0].id] == clusters[fragments[1].id]
    assert clusters[fragments[2].id] != clusters[fragments[0].id]


def test_strategies_respect_their_constraints(store, config):
    along = unit(0.45, (1 - 0.45**2) ** 0.5, 0)
    assert cosine_similarity(unit(1, 0, 0), along) == pytest.approx(0.45, abs=1e-5)
    _seed(store, "alpha always increases signal", "physics", unit(1, 0, 0))
    _seed(store, "beta never increases signal", "physics", along)
    _seed(store, "Local niches lose variety under selection pressure.", "biology", unit(0.2, 0.2, 0.96))
    _seed(store, "Alpha restates the same heat claim.", "physics", unit(1, 0, 0))
    fragments = store.all_fragments()

    bridge = select_pairs(fragments, "semantic_bridge", config, random.Random(0), 10)
    assert bridge
    assert all(item.strategy == "semantic_bridge" for item in bridge)
    clusters = assign_clusters(fragments, config.cluster_link_threshold)
    for pair in bridge:
        assert clusters[pair.fragment_a.id] != clusters[pair.fragment_b.id]
        assert config.min_similarity_for_candidate <= pair.similarity <= config.max_similarity_for_candidate
    scores = [item.selection_score for item in bridge]
    assert scores == sorted(scores, reverse=True)

    analogy = select_pairs(fragments, "analogical_transfer", config, random.Random(1), 10)
    assert analogy
    for pair in analogy:
        assert pair.fragment_a.domain == "biology"
        assert pair.fragment_b.domain == "physics"
        assert config.analogical_min_similarity <= pair.similarity <= config.analogical_max_similarity

    dialectic = select_pairs(fragments, "dialectic_synthesis", config, random.Random(2), 10)
    assert len(dialectic) == 1
    texts = {dialectic[0].fragment_a.text, dialectic[0].fragment_b.text}
    assert "alpha always increases signal" in texts
    assert "beta never increases signal" in texts


def test_exhaustive_selection_ignores_seed(store, config):
    _seed(store, "alpha always increases signal", "physics", unit(1, 0, 0))
    _seed(store, "beta never increases signal", "physics", unit(0.45, (1 - 0.45**2) ** 0.5, 0))
    _seed(store, "Local niches lose variety under selection pressure.", "biology", unit(0.2, 0.2, 0.96))
    fragments = store.all_fragments()
    first = select_pairs(fragments, "semantic_bridge", config, random.Random(0), 5)
    second = select_pairs(fragments, "semantic_bridge", config, random.Random(99), 5)
    assert [(item.fragment_a.id, item.fragment_b.id) for item in first] == [
        (item.fragment_a.id, item.fragment_b.id) for item in second
    ]
