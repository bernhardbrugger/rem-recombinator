"""
Diagnostic: samples random fragment pairs from the store and prints
the similarity distribution, so REM_MAX_SIM can be calibrated against
real data instead of guessed.

Usage:
    python diagnose_similarity.py
"""
import random

import numpy as np

from store import Store, cosine_similarity


def main():
    store = Store()
    fragments = store.all_fragments()
    print(f"{len(fragments)} fragments in store.\n")

    if len(fragments) < 2:
        print("Not enough fragments to compare.")
        return

    sims = []
    n_samples = min(2000, len(fragments) * (len(fragments) - 1) // 2)
    for _ in range(n_samples):
        a, b = random.sample(fragments, 2)
        sims.append(cosine_similarity(a.embedding, b.embedding))

    sims = np.array(sims)
    print(f"Similarity distribution over {n_samples} random pairs:")
    print(f"  min:    {sims.min():.3f}")
    print(f"  p10:    {np.percentile(sims, 10):.3f}")
    print(f"  p25:    {np.percentile(sims, 25):.3f}")
    print(f"  median: {np.percentile(sims, 50):.3f}")
    print(f"  p75:    {np.percentile(sims, 75):.3f}")
    print(f"  max:    {sims.max():.3f}")

    print(
        "\nSuggestion: set REM_MAX_SIM in .env to roughly the p10-p25 value "
        "above, so the recombinator only picks genuinely distant pairs "
        "(the bottom 10-25% of the similarity distribution) instead of an "
        "arbitrary fixed number."
    )


if __name__ == "__main__":
    main()
