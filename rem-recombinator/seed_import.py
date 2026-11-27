"""Compatibility shim. Prefer ``python -m rem_recombinator import``."""

import sys

from rem_recombinator.seed_import import (
    ImportReport,
    domain_from_memory_path,
    extract_fragments,
    load_export,
    run,
)

if __name__ == "__main__":
    from rem_recombinator.cli import main

    raise SystemExit(main(["import", *sys.argv[1:]]))
