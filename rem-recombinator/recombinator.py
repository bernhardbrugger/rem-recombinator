"""
The core REM mechanism: picks fragment pairs with low similarity
(= thematically distant) and asks an LLM for a non-obvious connection
between them.

Deliberately kept simple for the prototype: random sampling instead of
a clever pairing strategy. First idea for later optimization: weight
pairing toward "both not accessed in a long time" (an analogy to
memories that are preferentially reactivated during sleep).
"""
import random

import anthropic

import config
from store import Store, Fragment, cosine_similarity


RECOMBINATION_PROMPT = """You are given two knowledge fragments about the same \
person that appear unrelated at first glance.

Fragment A: {text_a}

Fragment B: {text_b}

Is there a non-obvious connection, pattern, or idea that emerges when you \
consider both together? Be concrete and practical — no vague platitudes \
("both show creativity"). If nothing meaningful genuinely comes to mind, say \
so honestly in one sentence instead of forcing something.

Answer in at most 3 sentences."""


def find_candidate_pairs(fragments: list[Fragment], n: int) -> list[tuple[Fragment, Fragment, float]]:
    """Samples random pairs, keeps the ones with lowest similarity."""
    if len(fragments) < 2:
        return []

    candidates = []
    attempts = min(n * 10, len(fragments) * (len(fragments) - 1) // 2)
    seen = set()

    for _ in range(attempts):
        a, b = random.sample(fragments, 2)
        key = tuple(sorted([a.id, b.id]))
        if key in seen:
            continue
        seen.add(key)

        sim = cosine_similarity(a.embedding, b.embedding)
        if sim < config.MAX_SIMILARITY_FOR_CANDIDATE:
            candidates.append((a, b, sim))

    # lowest similarity first = "most distant"
    candidates.sort(key=lambda x: x[2])
    return candidates[:n]


def generate_insight(client: anthropic.Anthropic, frag_a: Fragment, frag_b: Fragment) -> str:
    message = client.messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": RECOMBINATION_PROMPT.format(
                    text_a=frag_a.text, text_b=frag_b.text
                ),
            }
        ],
    )
    return message.content[0].text


def run_recombination_cycle() -> list[dict]:
    """One full REM cycle. Returns raw (unfiltered) results —
    filtering happens in critic.py."""
    config.validate()
    store = Store()
    fragments = store.all_fragments()

    if len(fragments) < 2:
        return []

    pairs = find_candidate_pairs(fragments, config.CANDIDATES_PER_RUN)
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    results = []
    for frag_a, frag_b, sim in pairs:
        insight = generate_insight(client, frag_a, frag_b)
        results.append(
            {
                "fragment_a": frag_a,
                "fragment_b": frag_b,
                "similarity": sim,
                "raw_insight": insight,
            }
        )
    return results
