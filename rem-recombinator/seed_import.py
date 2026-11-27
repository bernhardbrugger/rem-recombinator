"""
Imports your Claude memory export as seed fragments into your own store.

The export (Settings > Privacy > Export data, category "memories") is a
single JSON file with this shape:

    {
      "conversations_memory": "<one long markdown text>",
      "project_memories": {"<project_uuid>": "<free text>", ...},
      "memory_files": [
        {"path": "/areas/foo.md", "content": "<markdown with '- [stated]' lines>",
         "updated_at": "..."},
        ...
      ],
      "account_uuid": "..."
    }

The file may arrive either as raw .json or wrapped in a .zip — this
script handles both.

Fragment extraction strategy:
  - memory_files: each "- [stated] ..." bullet line becomes its own
    fragment (most atomic, highest quality — this is Claude's own
    curated memory).
  - project_memories: split into paragraphs (blank-line separated).
  - conversations_memory: split into paragraphs.

Usage:
    python seed_import.py /path/to/memories-export.json
    python seed_import.py /path/to/memories-export.json --dry-run
"""
import json
import re
import sys
import zipfile
from pathlib import Path

import config
from store import Store


def load_export(path: Path) -> dict:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            json_names = [n for n in zf.namelist() if n.endswith(".json")]
            if not json_names:
                raise ValueError(f"No .json file found inside {path}")
            return json.loads(zf.read(json_names[0]))
    return json.loads(path.read_text(encoding="utf-8"))


def _split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if len(p.strip()) > 20]


def extract_fragments(data: dict) -> list[dict]:
    fragments = []

    # 1. memory_files — atomic "- [stated] ..." bullet lines
    for mf in data.get("memory_files", []):
        path = mf.get("path", "unknown")
        for line in mf.get("content", "").splitlines():
            line = line.strip()
            if line.startswith("- [stated]") or line.startswith("-  [stated]"):
                text = line.split("]", 1)[1].strip() if "]" in line else line
                if len(text) > 10:
                    fragments.append(
                        {"text": text, "source": "memory_file", "origin": path}
                    )

    # 2. project_memories — free-text paragraphs per project
    for project_uuid, text in data.get("project_memories", {}).items():
        for para in _split_paragraphs(text):
            fragments.append(
                {"text": para, "source": "project_memory", "origin": project_uuid}
            )

    # 3. conversations_memory — free-text paragraphs
    conv_text = data.get("conversations_memory", "")
    for para in _split_paragraphs(conv_text):
        fragments.append(
            {"text": para, "source": "conversations_memory", "origin": "conversations_memory"}
        )

    return fragments


def run(export_path: str, dry_run: bool = False):
    path = Path(export_path)
    if not path.exists():
        raise FileNotFoundError(f"Export file not found: {path}")

    print(f"Reading export: {path}")
    data = load_export(path)
    fragments = extract_fragments(data)

    by_source = {}
    for f in fragments:
        by_source[f["source"]] = by_source.get(f["source"], 0) + 1

    print(f"Extracted {len(fragments)} fragments:")
    for source, count in by_source.items():
        print(f"  {source}: {count}")

    if dry_run:
        print("\n--dry-run set, skipping embedding and storage. Sample fragments:")
        for f in fragments[:5]:
            print(f"  [{f['source']}] {f['text'][:100]}")
        return

    config.validate()
    import embeddings  # imported here so --dry-run works without these deps installed

    store = Store()

    existing_texts = {f.text for f in store.all_fragments()}
    before = len(fragments)
    fragments = [f for f in fragments if f["text"] not in existing_texts]
    skipped = before - len(fragments)
    if skipped:
        print(f"Skipping {skipped} fragments already in the store (resume).")

    if not fragments:
        print("Nothing new to import.")
        return

    texts = [f["text"] for f in fragments]

    batch_size = 100
    added = 0
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vecs = embeddings.embed_batch(batch, input_type="document")
        for frag, vec in zip(fragments[i : i + batch_size], vecs):
            store.add_fragment(
                text=frag["text"],
                embedding=vec,
                source=frag["source"],
                metadata={"origin": frag["origin"]},
            )
            added += 1
        print(f"  {added}/{len(texts)} imported...")

    print(f"Done. {added} fragments in store: {store.db_path}")


if __name__ == "__main__":
    args = sys.argv[1:]
    dry_run = "--dry-run" in args
    args = [a for a in args if a != "--dry-run"]
    if len(args) != 1:
        print("Usage: python seed_import.py /path/to/memories-export.json [--dry-run]")
        sys.exit(1)
    run(args[0], dry_run=dry_run)
