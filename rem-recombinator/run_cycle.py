"""Compatibility shim for a single recombination run."""

import sys

from rem_recombinator.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["run", *sys.argv[1:]]))
