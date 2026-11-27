"""SQLite store: fragments, experiment runs, candidates, critiques, lineage.

Embeddings are float32 BLOBs. Similarity search is in-process, which is the
right trade for a few thousand fragments. The public methods are the seam a
future pgvector backend would implement.

Opening a database created by the first prototype adds ``domain`` and
``embedding_dim`` and leaves the old ``recombinations`` table readable.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np

from rem_recombinator.config import ExperimentConfig
from rem_recombinator.embeddings import as_vector, cosine_similarity
from rem_recombinator.models import CritiqueReport, Fragment, RecombinationCandidate

SCHEMA = """
CREATE TABLE IF NOT EXISTS fragments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    source TEXT NOT NULL,
    domain TEXT NOT NULL DEFAULT 'unspecified',
    embedding BLOB NOT NULL,
    embedding_dim INTEGER NOT NULL,
    created_at REAL NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS experiment_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    seed INTEGER NOT NULL,
    strategy TEXT NOT NULL,
    hyperparameters TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    strategy TEXT NOT NULL,
    hypothesis TEXT NOT NULL,
    rationale TEXT NOT NULL,
    pair_similarity REAL NOT NULL,
    novelty_score REAL,
    nearest_fragment_id INTEGER,
    nearest_similarity REAL,
    filter_status TEXT NOT NULL,
    parent_snapshot TEXT NOT NULL,
    selection_notes TEXT NOT NULL DEFAULT '',
    error TEXT,
    materialized_fragment_id INTEGER,
    created_at REAL NOT NULL,
    FOREIGN KEY (run_id) REFERENCES experiment_runs(id)
);

CREATE TABLE IF NOT EXISTS candidate_parents (
    candidate_id INTEGER NOT NULL,
    parent_id INTEGER NOT NULL,
    position INTEGER NOT NULL,
    PRIMARY KEY (candidate_id, parent_id),
    FOREIGN KEY (candidate_id) REFERENCES candidates(id),
    FOREIGN KEY (parent_id) REFERENCES fragments(id)
);

CREATE TABLE IF NOT EXISTS critiques (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id INTEGER NOT NULL UNIQUE,
    novelty REAL NOT NULL,
    logical_coherence REAL NOT NULL,
    falsifiability REAL NOT NULL,
    cross_domain_utility REAL NOT NULL,
    rationales TEXT NOT NULL,
    improvement TEXT NOT NULL,
    composite_score REAL NOT NULL,
    accept INTEGER NOT NULL,
    parse_error TEXT,
    created_at REAL NOT NULL,
    FOREIGN KEY (candidate_id) REFERENCES candidates(id)
);

CREATE TABLE IF NOT EXISTS lineage (
    child_id INTEGER NOT NULL,
    parent_id INTEGER NOT NULL,
    relation TEXT NOT NULL,
    PRIMARY KEY (child_id, parent_id),
    FOREIGN KEY (child_id) REFERENCES fragments(id),
    FOREIGN KEY (parent_id) REFERENCES fragments(id)
);

CREATE INDEX IF NOT EXISTS idx_candidates_run ON candidates(run_id);
CREATE INDEX IF NOT EXISTS idx_lineage_child ON lineage(child_id);
CREATE INDEX IF NOT EXISTS idx_lineage_parent ON lineage(parent_id);
"""

FILTER_STATUSES = frozenset(
    {"kept", "too_similar", "incoherent", "no_genuine_link", "error"}
)


class Store:
    """Single-writer SQLite connection."""

    def __init__(self, db_path: Path | str, clock=time.time) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._migrate()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def add_fragment(
        self,
        text: str,
        embedding: np.ndarray,
        source: str,
        domain: str = "unspecified",
        metadata: dict[str, Any] | None = None,
        created_at: float | None = None,
    ) -> int:
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("fragment text must not be empty")
        if not source.strip():
            raise ValueError("fragment source must not be empty")
        vector = as_vector(embedding)
        self._require_compatible_dim(int(vector.shape[0]))
        payload = metadata or {}
        if not isinstance(payload, dict):
            raise ValueError("metadata must be a JSON object")
        cur = self.conn.execute(
            """INSERT INTO fragments
               (text, source, domain, embedding, embedding_dim, created_at, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                cleaned,
                source.strip(),
                (domain or "unspecified").strip() or "unspecified",
                vector.tobytes(),
                int(vector.shape[0]),
                self._clock() if created_at is None else created_at,
                json.dumps(payload, sort_keys=True),
            ),
        )
        self.conn.commit()
        if cur.lastrowid is None:
            raise RuntimeError("SQLite did not return a fragment id")
        return int(cur.lastrowid)

    def get_fragment(self, fragment_id: int) -> Fragment:
        row = self.conn.execute(
            "SELECT * FROM fragments WHERE id = ?", (fragment_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"no fragment {fragment_id}")
        return _fragment_from_row(row)

    def all_fragments(self) -> list[Fragment]:
        rows = self.conn.execute(
            "SELECT * FROM fragments ORDER BY id ASC"
        ).fetchall()
        return [_fragment_from_row(row) for row in rows]

    def count_fragments(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) AS n FROM fragments").fetchone()["n"])

    def existing_texts(self) -> set[str]:
        rows = self.conn.execute("SELECT text FROM fragments").fetchall()
        return {row["text"] for row in rows}

    def start_run(self, config: ExperimentConfig, strategy: str) -> int:
        snapshot = config.public_dict()
        snapshot["strategy"] = strategy
        cur = self.conn.execute(
            """INSERT INTO experiment_runs (created_at, seed, strategy, hyperparameters)
               VALUES (?, ?, ?, ?)""",
            (
                self._clock(),
                config.seed,
                strategy,
                json.dumps(snapshot, sort_keys=True),
            ),
        )
        self.conn.commit()
        if cur.lastrowid is None:
            raise RuntimeError("SQLite did not return a run id")
        return int(cur.lastrowid)

    def save_candidate(
        self,
        *,
        run_id: int,
        strategy: str,
        hypothesis: str,
        rationale: str,
        pair_similarity: float,
        parent_ids: list[int],
        parent_snapshot: list[dict[str, Any]],
        selection_notes: str,
        filter_status: str,
        novelty_score: float | None = None,
        nearest_fragment_id: int | None = None,
        nearest_similarity: float | None = None,
        error: str | None = None,
    ) -> int:
        if filter_status not in FILTER_STATUSES:
            raise ValueError(f"unknown filter status {filter_status!r}")
        if len(parent_ids) < 2:
            raise ValueError("a recombination needs at least two parents")
        cur = self.conn.execute(
            """INSERT INTO candidates (
                   run_id, strategy, hypothesis, rationale, pair_similarity,
                   novelty_score, nearest_fragment_id, nearest_similarity,
                   filter_status, parent_snapshot, selection_notes, error, created_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                strategy,
                hypothesis,
                rationale,
                pair_similarity,
                novelty_score,
                nearest_fragment_id,
                nearest_similarity,
                filter_status,
                json.dumps(parent_snapshot, sort_keys=True),
                selection_notes,
                error,
                self._clock(),
            ),
        )
        if cur.lastrowid is None:
            raise RuntimeError("SQLite did not return a candidate id")
        candidate_id = int(cur.lastrowid)
        for position, parent_id in enumerate(parent_ids):
            self.conn.execute(
                """INSERT INTO candidate_parents (candidate_id, parent_id, position)
                   VALUES (?, ?, ?)""",
                (candidate_id, parent_id, position),
            )
        self.conn.commit()
        return candidate_id

    def save_critique(self, candidate_id: int, report: CritiqueReport) -> None:
        self.conn.execute(
            """INSERT INTO critiques (
                   candidate_id, novelty, logical_coherence, falsifiability,
                   cross_domain_utility, rationales, improvement, composite_score,
                   accept, parse_error, created_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                candidate_id,
                report.novelty,
                report.logical_coherence,
                report.falsifiability,
                report.cross_domain_utility,
                json.dumps(report.rationales, sort_keys=True),
                report.improvement,
                report.composite_score,
                int(report.accept),
                report.parse_error,
                self._clock(),
            ),
        )
        self.conn.commit()

    def materialize_candidate(self, candidate_id: int, embedding: np.ndarray) -> int:
        candidate = self.get_candidate(candidate_id)
        if candidate.materialized_fragment_id is not None:
            return candidate.materialized_fragment_id
        if candidate.filter_status != "kept" or candidate.critique is None or not candidate.critique.accept:
            raise ValueError("only an accepted, novelty-filtered candidate can be stored as a fragment")
        domains = [item.get("domain", "unspecified") for item in candidate.parent_snapshot]
        fragment_id = self.add_fragment(
            text=candidate.hypothesis,
            embedding=embedding,
            source="recombination",
            domain="recombination:" + "+".join(str(domain) for domain in domains),
            metadata={
                "candidate_id": candidate_id,
                "run_id": candidate.run_id,
                "strategy": candidate.strategy,
                "parent_ids": candidate.parent_ids,
            },
        )
        for parent_id in candidate.parent_ids:
            self.conn.execute(
                "INSERT INTO lineage (child_id, parent_id, relation) VALUES (?, ?, ?)",
                (fragment_id, parent_id, "recombination_parent"),
            )
        self.conn.execute(
            "UPDATE candidates SET materialized_fragment_id = ? WHERE id = ?",
            (fragment_id, candidate_id),
        )
        self.conn.commit()
        return fragment_id

    def get_candidate(self, candidate_id: int) -> RecombinationCandidate:
        row = self.conn.execute(
            "SELECT * FROM candidates WHERE id = ?", (candidate_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"no candidate {candidate_id}")
        return self._candidate_from_row(row)

    def candidates_for_run(self, run_id: int) -> list[RecombinationCandidate]:
        rows = self.conn.execute(
            "SELECT * FROM candidates WHERE run_id = ? ORDER BY id ASC", (run_id,)
        ).fetchall()
        return [self._candidate_from_row(row) for row in rows]

    def accepted_candidates(self, limit: int = 50) -> list[RecombinationCandidate]:
        rows = self.conn.execute(
            """SELECT c.* FROM candidates c
               JOIN critiques k ON k.candidate_id = c.id
               WHERE k.accept = 1
               ORDER BY c.created_at DESC, c.id DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [self._candidate_from_row(row) for row in rows]

    def fragment_lineage(self, fragment_id: int) -> dict[str, Any]:
        fragment = self.get_fragment(fragment_id)
        parents = self.conn.execute(
            """SELECT parent_id, relation FROM lineage
               WHERE child_id = ? ORDER BY parent_id ASC""",
            (fragment_id,),
        ).fetchall()
        children = self.conn.execute(
            """SELECT child_id, relation FROM lineage
               WHERE parent_id = ? ORDER BY child_id ASC""",
            (fragment_id,),
        ).fetchall()
        as_parent = self.conn.execute(
            """SELECT candidate_id, position FROM candidate_parents
               WHERE parent_id = ? ORDER BY candidate_id ASC""",
            (fragment_id,),
        ).fetchall()
        return {
            "fragment": _public_fragment(fragment),
            "parents": [
                {"id": int(row["parent_id"]), "relation": row["relation"]} for row in parents
            ],
            "children": [
                {"id": int(row["child_id"]), "relation": row["relation"]} for row in children
            ],
            "candidate_ids": [int(row["candidate_id"]) for row in as_parent],
        }

    def stats(self) -> dict[str, Any]:
        by_domain = {
            row["domain"]: int(row["n"])
            for row in self.conn.execute(
                "SELECT domain, COUNT(*) AS n FROM fragments GROUP BY domain ORDER BY domain"
            )
        }
        by_source = {
            row["source"]: int(row["n"])
            for row in self.conn.execute(
                "SELECT source, COUNT(*) AS n FROM fragments GROUP BY source ORDER BY source"
            )
        }
        by_status = {
            row["filter_status"]: int(row["n"])
            for row in self.conn.execute(
                "SELECT filter_status, COUNT(*) AS n FROM candidates GROUP BY filter_status ORDER BY filter_status"
            )
        }
        accepted = int(
            self.conn.execute("SELECT COUNT(*) AS n FROM critiques WHERE accept = 1").fetchone()["n"]
        )
        means: dict[str, float | None] = {}
        row = self.conn.execute(
            """SELECT AVG(novelty) AS novelty,
                      AVG(logical_coherence) AS logical_coherence,
                      AVG(falsifiability) AS falsifiability,
                      AVG(cross_domain_utility) AS cross_domain_utility,
                      AVG(composite_score) AS composite_score,
                      COUNT(*) AS n
               FROM critiques WHERE parse_error IS NULL"""
        ).fetchone()
        if row["n"]:
            means = {key: float(row[key]) for key in (
                "novelty",
                "logical_coherence",
                "falsifiability",
                "cross_domain_utility",
                "composite_score",
            )}
        return {
            "fragments": self.count_fragments(),
            "by_domain": by_domain,
            "by_source": by_source,
            "runs": int(self.conn.execute("SELECT COUNT(*) AS n FROM experiment_runs").fetchone()["n"]),
            "candidates": int(self.conn.execute("SELECT COUNT(*) AS n FROM candidates").fetchone()["n"]),
            "by_filter_status": by_status,
            "accepted": accepted,
            "mean_critique": means,
            "legacy_accepted": len(self.legacy_accepted()),
        }

    def legacy_accepted(self) -> list[dict[str, Any]]:
        if not self._has_table("recombinations"):
            return []
        rows = self.conn.execute(
            """SELECT id, fragment_id_a, fragment_id_b, similarity, raw_insight,
                      critic_score, critic_verdict, created_at
               FROM recombinations WHERE accepted = 1 ORDER BY created_at DESC"""
        ).fetchall()
        return [dict(row) for row in rows]

    def export_payload(self, *, accepted_only: bool = False) -> dict[str, Any]:
        runs = []
        for row in self.conn.execute(
            "SELECT * FROM experiment_runs ORDER BY id ASC"
        ).fetchall():
            candidates = self.candidates_for_run(int(row["id"]))
            if accepted_only:
                candidates = [item for item in candidates if item.critique and item.critique.accept]
            runs.append(
                {
                    "id": int(row["id"]),
                    "created_at": row["created_at"],
                    "seed": int(row["seed"]),
                    "strategy": row["strategy"],
                    "hyperparameters": json.loads(row["hyperparameters"]),
                    "candidates": [_public_candidate(item) for item in candidates],
                }
            )
        return {
            "runs": runs,
            "legacy_accepted": self.legacy_accepted(),
        }

    def runs(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT id, created_at, seed, strategy, hyperparameters FROM experiment_runs ORDER BY id ASC"
        ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["hyperparameters"] = json.loads(item["hyperparameters"])
            result.append(item)
        return result

    def _candidate_from_row(self, row: sqlite3.Row) -> RecombinationCandidate:
        parents = self.conn.execute(
            """SELECT parent_id FROM candidate_parents
               WHERE candidate_id = ? ORDER BY position ASC""",
            (row["id"],),
        ).fetchall()
        critique_row = self.conn.execute(
            "SELECT * FROM critiques WHERE candidate_id = ?", (row["id"],)
        ).fetchone()
        return RecombinationCandidate(
            id=int(row["id"]),
            run_id=int(row["run_id"]),
            parent_ids=[int(parent["parent_id"]) for parent in parents],
            strategy=row["strategy"],
            hypothesis=row["hypothesis"],
            rationale=row["rationale"],
            pair_similarity=float(row["pair_similarity"]),
            novelty_score=None if row["novelty_score"] is None else float(row["novelty_score"]),
            nearest_fragment_id=(
                None if row["nearest_fragment_id"] is None else int(row["nearest_fragment_id"])
            ),
            nearest_similarity=(
                None if row["nearest_similarity"] is None else float(row["nearest_similarity"])
            ),
            filter_status=row["filter_status"],
            selection_notes=row["selection_notes"],
            parent_snapshot=json.loads(row["parent_snapshot"]),
            critique=_critique_from_row(critique_row) if critique_row is not None else None,
            materialized_fragment_id=(
                None
                if row["materialized_fragment_id"] is None
                else int(row["materialized_fragment_id"])
            ),
            error=row["error"],
            created_at=float(row["created_at"]),
        )

    def _require_compatible_dim(self, dimensions: int) -> None:
        row = self.conn.execute(
            "SELECT embedding_dim FROM fragments WHERE embedding_dim != 0 LIMIT 1"
        ).fetchone()
        if row is not None and int(row["embedding_dim"]) != dimensions:
            raise ValueError(
                f"embedding dimension {dimensions} does not match stored width {int(row['embedding_dim'])}"
            )

    def _has_table(self, name: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
        ).fetchone()
        return row is not None

    def _migrate(self) -> None:
        existed = self._has_table("fragments")
        columns: set[str] = set()
        if existed:
            columns = {row["name"] for row in self.conn.execute("PRAGMA table_info(fragments)")}
        self.conn.executescript(SCHEMA)
        if existed:
            if "domain" not in columns:
                self.conn.execute(
                    "ALTER TABLE fragments ADD COLUMN domain TEXT NOT NULL DEFAULT 'unspecified'"
                )
                self.conn.execute(
                    """UPDATE fragments SET domain = source
                       WHERE domain = 'unspecified' AND source != ''"""
                )
            if "embedding_dim" not in columns:
                self.conn.execute(
                    "ALTER TABLE fragments ADD COLUMN embedding_dim INTEGER NOT NULL DEFAULT 0"
                )
            rows = self.conn.execute(
                "SELECT id, embedding FROM fragments WHERE embedding_dim = 0"
            ).fetchall()
            for row in rows:
                blob = row["embedding"]
                self.conn.execute(
                    "UPDATE fragments SET embedding_dim = ? WHERE id = ?",
                    (len(blob) // 4, row["id"]),
                )
        self.conn.commit()


def _fragment_from_row(row: sqlite3.Row) -> Fragment:
    embedding = np.frombuffer(row["embedding"], dtype=np.float32).copy()
    return Fragment(
        id=int(row["id"]),
        text=row["text"],
        source=row["source"],
        domain=row["domain"],
        created_at=float(row["created_at"]),
        embedding=embedding,
        metadata=json.loads(row["metadata"]),
        embedding_dim=int(row["embedding_dim"]),
    )


def _critique_from_row(row: sqlite3.Row) -> CritiqueReport:
    return CritiqueReport(
        novelty=float(row["novelty"]),
        logical_coherence=float(row["logical_coherence"]),
        falsifiability=float(row["falsifiability"]),
        cross_domain_utility=float(row["cross_domain_utility"]),
        rationales=json.loads(row["rationales"]),
        improvement=row["improvement"],
        composite_score=float(row["composite_score"]),
        accept=bool(row["accept"]),
        parse_error=row["parse_error"],
    )


def _public_fragment(fragment: Fragment) -> dict[str, Any]:
    return {
        "id": fragment.id,
        "text": fragment.text,
        "source": fragment.source,
        "domain": fragment.domain,
        "created_at": fragment.created_at,
        "embedding_dim": fragment.embedding_dim,
        "metadata": fragment.metadata,
    }


def _public_candidate(candidate: RecombinationCandidate) -> dict[str, Any]:
    critique = None
    if candidate.critique is not None:
        critique = {
            "novelty": candidate.critique.novelty,
            "logical_coherence": candidate.critique.logical_coherence,
            "falsifiability": candidate.critique.falsifiability,
            "cross_domain_utility": candidate.critique.cross_domain_utility,
            "rationales": candidate.critique.rationales,
            "improvement": candidate.critique.improvement,
            "composite_score": candidate.critique.composite_score,
            "accept": candidate.critique.accept,
            "parse_error": candidate.critique.parse_error,
        }
    return {
        "id": candidate.id,
        "run_id": candidate.run_id,
        "parent_ids": candidate.parent_ids,
        "parent_snapshot": candidate.parent_snapshot,
        "strategy": candidate.strategy,
        "hypothesis": candidate.hypothesis,
        "rationale": candidate.rationale,
        "pair_similarity": candidate.pair_similarity,
        "novelty_score": candidate.novelty_score,
        "nearest_fragment_id": candidate.nearest_fragment_id,
        "nearest_similarity": candidate.nearest_similarity,
        "filter_status": candidate.filter_status,
        "selection_notes": candidate.selection_notes,
        "materialized_fragment_id": candidate.materialized_fragment_id,
        "error": candidate.error,
        "created_at": candidate.created_at,
        "critique": critique,
    }


# Re-export so older imports of cosine_similarity from the store keep a single definition.
__all__ = ["Store", "Fragment", "cosine_similarity", "SCHEMA"]
