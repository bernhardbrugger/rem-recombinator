#!/usr/bin/env python3
"""Write docs/geometry.json from the fictional sample. No network."""

from __future__ import annotations

import json
from pathlib import Path

from rem_recombinator.public_corpus import build_public_geometry


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    target = root / "docs" / "geometry.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(build_public_geometry(), indent=2) + "\n", encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
