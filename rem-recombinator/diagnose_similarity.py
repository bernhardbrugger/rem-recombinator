"""Compatibility shim for the similarity diagnostic."""

import sys

from rem_recombinator.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["diagnose", *sys.argv[1:]]))
