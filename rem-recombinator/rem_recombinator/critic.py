"""Multi-axis critic.

The model fills a fixed schema. This module computes the composite score and
the accept decision, so two runs with the same dimension scores accept the
same hypotheses.

Rubric, each axis in ``[0, 1]``:

- novelty: 0 paraphrases a parent; 0.5 adds a minor qualification; 1 is not
  entailed by either parent alone.
- logical_coherence: 0 is a non sequitur or self-contradiction; 0.5 has a gap
  in the argument; 1 follows from the stated premises.
- falsifiability: 0 names no observation; 0.5 names a vague consequence; 1
  names an observation that would refute the hypothesis.
- cross_domain_utility: 0 is stuck in the source wording; 0.5 is useful in one
  neighbouring setting; 1 suggests a concrete transfer or experiment elsewhere.

Accept when the weighted composite is at least ``critic_accept_threshold`` and
every axis is at least ``critic_dimension_floor``.
"""

from __future__ import annotations

import json
import re
from typing import Any

from rem_recombinator.config import ExperimentConfig
from rem_recombinator.models import CritiqueReport

DIMENSIONS: tuple[str, ...] = (
    "novelty",
    "logical_coherence",
    "falsifiability",
    "cross_domain_utility",
)

CRITIC_RUBRIC = """Score each axis from 0 to 1.
novelty: 0 paraphrases a parent, 0.5 adds a minor qualification, 1 is not entailed by either parent alone.
logical_coherence: 0 is a non sequitur or self-contradiction, 0.5 has a reasoning gap, 1 follows from the stated premises.
falsifiability: 0 names no observation, 0.5 names a vague consequence, 1 names an observation that would refute the hypothesis.
cross_domain_utility: 0 stays inside the source wording, 0.5 is useful next door, 1 suggests a concrete transfer or experiment in another domain.
Write one sentence of rationale per axis and one improvement that would raise the weakest axis.
Do not decide acceptance. The pipeline does that from the scores."""


def composite_score(scores: dict[str, float], weights: dict[str, float]) -> float:
    return float(sum(scores[name] * weights[name] for name in DIMENSIONS))


def decide_accept(
    scores: dict[str, float],
    combined: float,
    *,
    threshold: float,
    floor: float,
) -> bool:
    return combined >= threshold and all(scores[name] >= floor for name in DIMENSIONS)


def parse_critique(payload: dict[str, Any] | str, config: ExperimentConfig) -> CritiqueReport:
    """Validate a model payload. Invalid payloads become a rejected report."""
    data: Any = payload
    if isinstance(payload, str):
        try:
            data = json.loads(_strip_fences(payload))
        except json.JSONDecodeError as exc:
            return _failed(f"critique JSON did not parse: {exc}")
    if not isinstance(data, dict):
        return _failed("critique payload must be a JSON object")
    try:
        scores = {name: _unit_score(data[name], name) for name in DIMENSIONS}
        rationales_raw = data.get("rationales")
        if not isinstance(rationales_raw, dict):
            raise ValueError("rationales must be an object")
        rationales = {name: _sentence(rationales_raw[name], name) for name in DIMENSIONS}
        improvement = _sentence(data.get("improvement"), "improvement")
    except (KeyError, TypeError, ValueError) as exc:
        return _failed(str(exc))
    weights = config.weights()
    combined = composite_score(scores, weights)
    return CritiqueReport(
        novelty=scores["novelty"],
        logical_coherence=scores["logical_coherence"],
        falsifiability=scores["falsifiability"],
        cross_domain_utility=scores["cross_domain_utility"],
        rationales=rationales,
        improvement=improvement,
        composite_score=combined,
        accept=decide_accept(
            scores,
            combined,
            threshold=config.critic_accept_threshold,
            floor=config.critic_dimension_floor,
        ),
    )


def _unit_score(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number in [0, 1]")
    score = float(value)
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"{name} must be inside [0, 1], got {score}")
    return score


def _sentence(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} rationale must be a non-empty string")
    return value.strip()


def _failed(message: str) -> CritiqueReport:
    return CritiqueReport(
        novelty=0.0,
        logical_coherence=0.0,
        falsifiability=0.0,
        cross_domain_utility=0.0,
        rationales={name: "" for name in DIMENSIONS},
        improvement="",
        composite_score=0.0,
        accept=False,
        parse_error=message,
    )


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
    stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()
