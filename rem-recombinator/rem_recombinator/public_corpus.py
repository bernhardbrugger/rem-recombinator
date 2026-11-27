"""Geometry of the fictional public sample.

The numbers on the GitHub Pages demonstration are produced by this function
and committed as ``docs/geometry.json``. The embedder is the offline signed
feature hash. A principal-component plane is a picture of that hash, not a
claim about meaning.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from rem_recombinator.config import ExperimentConfig
from rem_recombinator.embeddings import DeterministicEmbedder, cosine_similarity, unit_vector
from rem_recombinator.models import Fragment
from rem_recombinator.seed_import import load_fragment_records
from rem_recombinator.strategies import assign_clusters, select_pairs

SAMPLE = Path(__file__).resolve().parent.parent / "examples" / "sample_memory_export.json"


def build_public_geometry() -> dict:
    """Return the JSON object the public page draws."""
    config = ExperimentConfig()
    records = load_fragment_records(SAMPLE)
    embedder = DeterministicEmbedder()
    vectors = embedder.embed_batch([record["text"] for record in records], input_type="document")
    fragments = [
        Fragment(
            id=index,
            text=record["text"],
            source=record["source"],
            domain=record["domain"],
            created_at=0.0,
            embedding=vector,
        )
        for index, (record, vector) in enumerate(zip(records, vectors), start=1)
    ]
    matrix = np.stack([unit_vector(fragment.embedding).astype(np.float64) for fragment in fragments])
    centered = matrix - matrix.mean(axis=0)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    plane = centered @ vt[:2].T
    span = np.max(np.abs(plane), axis=0)
    span[span == 0.0] = 1.0
    scaled = plane / span
    variance = singular**2
    total = float(variance.sum())
    explained = [float(value / total) if total else 0.0 for value in variance[:2]]
    clusters = assign_clusters(fragments, config.cluster_link_threshold)
    rng = random.Random(0)
    selections = {
        name: select_pairs(fragments, name, config, rng, 3)
        for name in ("semantic_bridge", "analogical_transfer", "dialectic_synthesis")
    }
    return {
        "corpus": "fictional-sample",
        "embedder": "hashing-32",
        "disclaimer": (
            "Signed feature hash of seven fictional sentences, projected onto "
            "its first two principal components. Not a semantic embedding."
        ),
        "explained_variance": [_round(value) for value in explained],
        "thresholds": {
            "bridge": [config.min_similarity_for_candidate, config.max_similarity_for_candidate],
            "analogy": [config.analogical_min_similarity, config.analogical_max_similarity],
            "dialectic": [config.dialectic_min_similarity, config.dialectic_max_similarity],
            "dialectic_target": config.dialectic_target_similarity,
            "cluster_link": config.cluster_link_threshold,
            "novelty_ceiling": config.novelty_ceiling,
            "incoherence_floor": config.incoherence_floor,
            "accept": config.critic_accept_threshold,
            "axis_floor": config.critic_dimension_floor,
            "weights": config.weights(),
        },
        "fragments": [
            {
                "id": fragment.id,
                "domain": fragment.domain,
                "source": fragment.source,
                "text": fragment.text,
                "cluster": clusters[fragment.id],
                "x": _round(float(scaled[index, 0])),
                "y": _round(float(scaled[index, 1])),
            }
            for index, fragment in enumerate(fragments)
        ],
        "pairs": [
            {
                "a": fragments[i].id,
                "b": fragments[j].id,
                "cosine": _round(cosine_similarity(fragments[i].embedding, fragments[j].embedding)),
            }
            for i in range(len(fragments))
            for j in range(i + 1, len(fragments))
        ],
        "selected": {
            name: [
                {
                    "a": pair.fragment_a.id,
                    "b": pair.fragment_b.id,
                    "cosine": _round(pair.similarity),
                    "score": _round(pair.selection_score),
                }
                for pair in pairs
            ]
            for name, pairs in selections.items()
        },
    }


def _round(value: float) -> float:
    return round(float(value), 4)
