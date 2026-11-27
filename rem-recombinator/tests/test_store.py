"""Persistence, migration, and lineage."""

import json
import sqlite3

import numpy as np

from rem_recombinator.models import CritiqueReport
from rem_recombinator.store import Store
from tests.fakes import unit


def _critique(accept: bool = True) -> CritiqueReport:
    score = 0.8 if accept else 0.1
    return CritiqueReport(
        novelty=score,
        logical_coherence=score,
        falsifiability=score,
        cross_domain_utility=score,
        rationales={
            "novelty": "Not a paraphrase.",
            "logical_coherence": "The step follows.",
            "falsifiability": "A measurement could refute it.",
            "cross_domain_utility": "It suggests a transfer.",
        },
        improvement="Name the instrument.",
        composite_score=score,
        accept=accept,
    )


def test_round_trip_fragment_and_dimension_guard(store):
    first = store.add_fragment("alpha", unit(1, 0, 0), source="test", domain="physics")
    loaded = store.get_fragment(first)
    assert loaded.domain == "physics"
    assert loaded.embedding_dim == 3
    assert np.allclose(loaded.embedding, unit(1, 0, 0))
    try:
        store.add_fragment("beta", unit(1, 0), source="test", domain="physics")
    except ValueError as exc:
        assert "dimension" in str(exc)
    else:
        raise AssertionError("mixed embedding widths must be rejected")


def test_candidate_lineage_and_export(store, config):
    left = store.add_fragment("left claim", unit(1, 0, 0), source="test", domain="physics")
    right = store.add_fragment("right claim", unit(0, 1, 0), source="test", domain="biology")
    run_id = store.start_run(config, "semantic_bridge")
    candidate_id = store.save_candidate(
        run_id=run_id,
        strategy="semantic_bridge",
        hypothesis="A bridge claim.",
        rationale="Because the midpoint is empty.",
        pair_similarity=0.2,
        parent_ids=[left, right],
        parent_snapshot=[
            {"id": left, "domain": "physics", "source": "test", "text": "left claim"},
            {"id": right, "domain": "biology", "source": "test", "text": "right claim"},
        ],
        selection_notes="notes",
        filter_status="kept",
        novelty_score=0.5,
        nearest_fragment_id=left,
        nearest_similarity=0.5,
    )
    store.save_critique(candidate_id, _critique(True))
    child = store.materialize_candidate(candidate_id, unit(0.2, 0.2, 0.96))
    lineage = store.fragment_lineage(child)
    assert [item["id"] for item in lineage["parents"]] == [left, right]
    assert store.fragment_lineage(left)["children"][0]["id"] == child
    payload = store.export_payload()
    blob = json.dumps(payload)
    assert "A bridge claim." in blob
    assert "sk-ant" not in blob
    assert "anthropic_api_key" not in payload["runs"][0]["hyperparameters"]
    assert store.stats()["accepted"] == 1


def test_prototype_database_is_migrated(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE fragments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            source TEXT NOT NULL,
            embedding BLOB NOT NULL,
            created_at REAL NOT NULL,
            metadata TEXT NOT NULL DEFAULT '{}'
        )"""
    )
    blob = np.array([1, 0, 0], dtype=np.float32).tobytes()
    conn.execute(
        "INSERT INTO fragments (text, source, embedding, created_at) VALUES (?, ?, ?, ?)",
        ("legacy", "memory_file", blob, 1.0),
    )
    conn.execute(
        """CREATE TABLE recombinations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fragment_id_a INTEGER NOT NULL,
            fragment_id_b INTEGER NOT NULL,
            similarity REAL NOT NULL,
            raw_insight TEXT NOT NULL,
            critic_score REAL,
            critic_verdict TEXT,
            accepted INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL
        )"""
    )
    conn.execute(
        """INSERT INTO recombinations
           (fragment_id_a, fragment_id_b, similarity, raw_insight, critic_score,
            critic_verdict, accepted, created_at)
           VALUES (1, 1, 0.1, 'old insight', 0.9, 'ok', 1, 2.0)"""
    )
    conn.commit()
    conn.close()

    store = Store(path)
    fragment = store.get_fragment(1)
    assert fragment.domain == "memory_file"
    assert fragment.embedding_dim == 3
    assert store.legacy_accepted()[0]["raw_insight"] == "old insight"
    store.close()
    again = Store(path)
    assert again.get_fragment(1).domain == "memory_file"
    again.close()
