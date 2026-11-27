"""
Second LLM call acting as a quality filter. Most random recombinations
are noise (no different from real REM sleep) — the critic decides what
actually gets surfaced to the user.

Deliberately a SEPARATE call (not part of the recombination prompt):
the model shouldn't justify its own idea in the same context window
where it generated it — that favors confirmation over genuine critique.
"""
import json

import anthropic

import config
from store import Store


CRITIC_PROMPT = """Another system generated the following possible connection \
from two independent knowledge fragments about a person:

Fragment A: {text_a}
Fragment B: {text_b}
Generated connection: {insight}

Evaluate ONLY whether this is a genuine, useful, non-trivial insight — not \
whether it sounds nice. Be strict: most auto-generated connections are random \
noise or platitudes.

Answer ONLY as JSON, no further explanation:
{{"score": <0.0-1.0>, "verdict": "<one sentence justification>", "accept": <true/false>}}

accept=true only if score >= 0.6 AND the connection is actually actionable \
or surprising."""


def evaluate(client: anthropic.Anthropic, text_a: str, text_b: str, insight: str) -> dict:
    message = client.messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=200,
        messages=[
            {
                "role": "user",
                "content": CRITIC_PROMPT.format(
                    text_a=text_a, text_b=text_b, insight=insight
                ),
            }
        ],
    )
    raw = message.content[0].text.strip()
    # Robust against markdown fences, in case the model adds them anyway
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"score": 0.0, "verdict": "Critic response not parsable", "accept": False}


def filter_results(raw_results: list[dict]) -> list[dict]:
    """Takes the raw recombinator results, evaluates each one, stores
    all of them (including rejected ones, for later analysis), and
    returns only the accepted ones."""
    config.validate()
    store = Store()
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    accepted = []
    for r in raw_results:
        verdict = evaluate(
            client, r["fragment_a"].text, r["fragment_b"].text, r["raw_insight"]
        )
        store.save_recombination(
            fragment_id_a=r["fragment_a"].id,
            fragment_id_b=r["fragment_b"].id,
            similarity=r["similarity"],
            raw_insight=r["raw_insight"],
            critic_score=verdict.get("score"),
            critic_verdict=verdict.get("verdict"),
            accepted=verdict.get("accept", False),
        )
        if verdict.get("accept"):
            accepted.append({**r, **verdict})

    return accepted
