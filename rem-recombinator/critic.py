"""Compatibility shim. The implementation is :mod:`rem_recombinator.critic`."""

from rem_recombinator.critic import (
    CRITIC_RUBRIC,
    DIMENSIONS,
    composite_score,
    decide_accept,
    parse_critique,
)

__all__ = [
    "CRITIC_RUBRIC",
    "DIMENSIONS",
    "composite_score",
    "decide_accept",
    "parse_critique",
]
