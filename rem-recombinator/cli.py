"""Compatibility shim. ``python cli.py`` and ``python -m rem_recombinator``."""

import sys

from rem_recombinator.cli import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
