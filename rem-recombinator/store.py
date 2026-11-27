"""
Storage layer for knowledge fragments.

MVP decision: SQLite instead of Postgres/a dedicated vector DB.
Embeddings are stored as BLOBs (numpy float32); similarity search runs
in-memory (numpy). Fast enough for a few thousand fragments — no reason
to add infra overhead at the prototype stage.

If the dataset grows: swap the same interface methods against pgvector
on the Hetzner instance, without touching recombinator.py.
"""
import sqlite3
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

import config


@dataclass
class Fragment:
    id: int
    text: str
    source: str          # e.g. "claude_memory_export", "manual"
    embedding: np.ndarray
    created_at: float
    metadata: dict


SCHEMA = """
CREATE TABLE IF NOT EXISTS fragments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    source TEXT NOT NULL,
    embedding BLOB NOT NULL,
    created_at REAL NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS recombinations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fragment_id_a INTEGER NOT NULL,
    fragment_id_b INTEGER NOT NULL,
    similarity REAL NOT NULL,
    raw_insight TEXT NOT NULL,
    critic_score REAL,
    critic_verdict TEXT,
    accepted INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    FOREIGN KEY (fragment_id_a) REFERENCES fragments(id),
    FOREIGN KEY (fragment_id_b) REFERENCES fragments(id)
);
"""


class Store:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or config.DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def add_fragment(
        self, text: str, embedding: np.ndarray, source: str, metadata: dict | None = None
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO fragments (text, source, embedding, created_at, metadata) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                text,
                source,
                embedding.astype(np.float32).tobytes(),
                time.time(),
                json.dumps(metadata or {}),
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    def all_fragments(self) -> list[Fragment]:
        rows = self.conn.execute(
            "SELECT id, text, source, embedding, created_at, metadata FROM fragments"
        ).fetchall()
        return [
            Fragment(
                id=r[0],
                text=r[1],
                source=r[2],
                embedding=np.frombuffer(r[3], dtype=np.float32),
                created_at=r[4],
                metadata=json.loads(r[5]),
            )
            for r in rows
        ]

    def count_fragments(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM fragments").fetchone()[0]

    def save_recombination(
        self,
        fragment_id_a: int,
        fragment_id_b: int,
        similarity: float,
        raw_insight: str,
        critic_score: float | None = None,
        critic_verdict: str | None = None,
        accepted: bool = False,
    ) -> int:
        cur = self.conn.execute(
            """INSERT INTO recombinations
               (fragment_id_a, fragment_id_b, similarity, raw_insight,
                critic_score, critic_verdict, accepted, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                fragment_id_a,
                fragment_id_b,
                similarity,
                raw_insight,
                critic_score,
                critic_verdict,
                int(accepted),
                time.time(),
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    def accepted_recombinations(self) -> list[dict]:
        rows = self.conn.execute(
            """SELECT r.id, fa.text, fb.text, r.raw_insight, r.critic_score, r.created_at
               FROM recombinations r
               JOIN fragments fa ON fa.id = r.fragment_id_a
               JOIN fragments fb ON fb.id = r.fragment_id_b
               WHERE r.accepted = 1
               ORDER BY r.created_at DESC"""
        ).fetchall()
        return [
            {
                "id": r[0],
                "fragment_a": r[1],
                "fragment_b": r[2],
                "insight": r[3],
                "critic_score": r[4],
                "created_at": r[5],
            }
            for r in rows
        ]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
