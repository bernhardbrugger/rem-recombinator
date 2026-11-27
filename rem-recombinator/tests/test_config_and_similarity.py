"""Configuration snapshots and cosine geometry."""

import json

import numpy as np
import pytest

from rem_recombinator.config import ConfigurationError, load_config
from rem_recombinator.embeddings import DeterministicEmbedder, cosine_similarity


def test_public_dict_omits_secrets():
    config = load_config(
        {"ANTHROPIC_API_KEY": "sk-ant-example", "VOYAGE_API_KEY": "pa-examplekey"}
    )
    blob = json.dumps(config.public_dict())
    assert "sk-ant" not in blob
    assert "pa-examplekey" not in blob
    assert "anthropic_api_key" not in blob
    assert config.public_dict()["seed"] == 0
    assert config.public_dict()["anthropic_model"] == "claude-sonnet-4-6"


def test_weights_must_sum_to_one():
    with pytest.raises(ConfigurationError, match="sum to 1"):
        load_config({"REM_WEIGHT_NOVELTY": "0.9"})


def test_bad_float_names_the_variable():
    with pytest.raises(ConfigurationError, match="REM_MAX_SIM"):
        load_config({"REM_MAX_SIM": "high"})


def test_cosine_identity_orthogonal_and_zero():
    axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    other = np.array([0.0, 2.0, 0.0], dtype=np.float32)
    assert cosine_similarity(axis, axis) == pytest.approx(1.0)
    assert cosine_similarity(axis, other) == pytest.approx(0.0)
    assert cosine_similarity(axis, np.zeros(3, dtype=np.float32)) == 0.0
    with pytest.raises(ValueError, match="dimension"):
        cosine_similarity(axis, np.ones(4, dtype=np.float32))


def test_hashing_embedder_is_deterministic_and_token_sensitive():
    embedder = DeterministicEmbedder(dimensions=32)
    first = embedder.embed_text("entropy increases")
    assert np.array_equal(first, embedder.embed_text("entropy increases"))
    assert cosine_similarity(first, embedder.embed_text("entropy increases today")) > cosine_similarity(
        first, embedder.embed_text("unrelated lattice tiling")
    )
    assert first.shape == (32,)
    assert np.linalg.norm(first) == pytest.approx(1.0, abs=1e-5)
