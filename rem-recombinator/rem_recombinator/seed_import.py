"""Import a Claude memory export into the fragment store.

The export shape (Settings, Privacy, Export data, category "memories") is::

    {
      "conversations_memory": "<markdown>",
      "project_memories": {"<uuid>": "<text>"},
      "memory_files": [{"path": "/areas/foo.md", "content": "- [stated] ..."}],
      "account_uuid": "..."
    }

The file may be raw JSON or a zip that contains one JSON file. Extraction is
pure; embedding and storage happen only when an embedder and a store are passed.
"""

from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from rem_recombinator.embeddings import Embedder
from rem_recombinator.store import Store


@dataclass
class ImportReport:
    extracted: int
    added: int
    skipped_existing: int
    by_source: dict[str, int] = field(default_factory=dict)
    samples: list[dict[str, str]] = field(default_factory=list)


def load_export(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"export file not found: {path}")
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = [name for name in archive.namelist() if name.endswith(".json")]
            if not names:
                raise ValueError(f"no .json file found inside {path}")
            return json.loads(archive.read(names[0]))
    return json.loads(path.read_text(encoding="utf-8"))


def extract_fragments(data: dict) -> list[dict[str, str]]:
    """Turn an export object into fragment dicts with text, source, domain, origin."""
    if not isinstance(data, dict):
        raise ValueError("memory export must be a JSON object")
    fragments: list[dict[str, str]] = []
    for memory_file in data.get("memory_files") or []:
        if not isinstance(memory_file, dict):
            continue
        path = str(memory_file.get("path") or "unknown")
        domain = domain_from_memory_path(path)
        for line in str(memory_file.get("content") or "").splitlines():
            stripped = line.strip()
            if stripped.startswith("- [stated]") or stripped.startswith("-  [stated]"):
                text = stripped.split("]", 1)[1].strip() if "]" in stripped else stripped
                if len(text) > 10:
                    fragments.append(
                        {
                            "text": text,
                            "source": "memory_file",
                            "domain": domain,
                            "origin": path,
                        }
                    )
    for project_id, text in (data.get("project_memories") or {}).items():
        for paragraph in _split_paragraphs(str(text)):
            fragments.append(
                {
                    "text": paragraph,
                    "source": "project_memory",
                    "domain": "project",
                    "origin": str(project_id),
                }
            )
    for paragraph in _split_paragraphs(str(data.get("conversations_memory") or "")):
        fragments.append(
            {
                "text": paragraph,
                "source": "conversations_memory",
                "domain": "conversation",
                "origin": "conversations_memory",
            }
        )
    return fragments


def domain_from_memory_path(path: str) -> str:
    """Use the first directory of a memory path as the domain tag."""
    parts = [part for part in path.strip("/").split("/") if part]
    if not parts:
        return "memory"
    if len(parts) == 1 and parts[0].endswith(".md"):
        return "memory"
    return parts[0]


def import_extracted(
    fragments: list[dict[str, str]],
    store: Store,
    embedder: Embedder,
    *,
    batch_size: int = 100,
) -> ImportReport:
    """Embed and insert fragments whose text is not already stored."""
    normalized = [{**fragment, "text": fragment["text"].strip()} for fragment in fragments]
    by_source: dict[str, int] = {}
    for fragment in normalized:
        by_source[fragment["source"]] = by_source.get(fragment["source"], 0) + 1
    existing = store.existing_texts()
    pending = [fragment for fragment in normalized if fragment["text"] not in existing]
    added = 0
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        vectors = embedder.embed_batch([item["text"] for item in batch], input_type="document")
        for fragment, vector in zip(batch, vectors):
            store.add_fragment(
                text=fragment["text"],
                embedding=vector,
                source=fragment["source"],
                domain=fragment["domain"],
                metadata={"origin": fragment["origin"]},
            )
            added += 1
    return ImportReport(
        extracted=len(fragments),
        added=added,
        skipped_existing=len(fragments) - len(pending),
        by_source=by_source,
        samples=[{"source": item["source"], "text": item["text"][:160]} for item in fragments[:5]],
    )


def run(
    export_path: str | Path,
    store: Store | None = None,
    embedder: Embedder | None = None,
    *,
    dry_run: bool = False,
) -> ImportReport:
    data = load_export(Path(export_path))
    fragments = extract_fragments(data)
    if dry_run:
        by_source: dict[str, int] = {}
        for fragment in fragments:
            by_source[fragment["source"]] = by_source.get(fragment["source"], 0) + 1
        return ImportReport(
            extracted=len(fragments),
            added=0,
            skipped_existing=0,
            by_source=by_source,
            samples=[{"source": item["source"], "text": item["text"][:160]} for item in fragments[:5]],
        )
    if store is None or embedder is None:
        raise ValueError("import requires a store and an embedder")
    return import_extracted(fragments, store, embedder)


def _split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", text)
    return [part.strip() for part in parts if len(part.strip()) > 20]
