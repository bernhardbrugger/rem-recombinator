"""
Quick manual test for a single REM cycle — run this directly instead
of wrestling with shell quoting for one-liners.

Usage:
    python run_cycle.py
"""
from critic import filter_results
from recombinator import run_recombination_cycle


def main():
    print("Running recombination cycle...")
    raw = run_recombination_cycle()
    print(f"{len(raw)} candidate pairs generated.\n")

    if not raw:
        print("Not enough fragments in the store.")
        return

    print("Running critic filter...")
    accepted = filter_results(raw)
    print(f"\n{len(accepted)} of {len(raw)} accepted:\n")

    for r in accepted:
        print(f"- {r['raw_insight']}")
        print(f"  (score: {r['score']:.2f}, verdict: {r['verdict']})\n")

    if not accepted:
        print("None passed the critic filter this round — try running again,")
        print("or check the 'recombinations' table in data/fragments.db for")
        print("the raw (rejected) attempts.")


if __name__ == "__main__":
    main()
