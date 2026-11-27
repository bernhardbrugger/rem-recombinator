"""Language-model boundary.

The pipeline depends on ``recombine`` and ``critique``, not on a vendor SDK.
``AnthropicLanguageModel`` forces a tool call so the payload matches the
schema. ``ScriptedLanguageModel`` and ``TemplateLanguageModel`` keep tests and
offline demos off the network. The template model does not evaluate ideas.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Protocol

from rem_recombinator.config import ExperimentConfig
from rem_recombinator.critic import CRITIC_RUBRIC, DIMENSIONS

logger = logging.getLogger(__name__)

STRATEGY_INSTRUCTIONS: dict[str, str] = {
    "semantic_bridge": (
        "Semantic bridge: propose a hypothesis that would occupy the unoccupied "
        "region between the two fragments. Name a mechanism. If no coherent "
        "bridge exists, set no_genuine_link to true and explain why in rationale."
    ),
    "analogical_transfer": (
        "Analogical transfer: map a relational structure from the source domain "
        "onto the target domain. Name the mapped relations. Do not claim the "
        "mapping is empirically established. If the domains share no relational "
        "structure, set no_genuine_link to true."
    ),
    "dialectic_synthesis": (
        "Dialectic synthesis: the fragments are in tension. Propose a hypothesis "
        "that resolves the tension or states it as a precise disagreement. Say "
        "what observation would support it and what would count against it. If "
        "the texts do not actually conflict, set no_genuine_link to true."
    ),
}

_HYPOTHESIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "hypothesis": {"type": "string"},
        "rationale": {"type": "string"},
        "no_genuine_link": {"type": "boolean"},
    },
    "required": ["hypothesis", "rationale", "no_genuine_link"],
}

_CRITIQUE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "novelty": {"type": "number", "minimum": 0, "maximum": 1},
        "logical_coherence": {"type": "number", "minimum": 0, "maximum": 1},
        "falsifiability": {"type": "number", "minimum": 0, "maximum": 1},
        "cross_domain_utility": {"type": "number", "minimum": 0, "maximum": 1},
        "rationales": {
            "type": "object",
            "properties": {name: {"type": "string"} for name in DIMENSIONS},
            "required": list(DIMENSIONS),
        },
        "improvement": {"type": "string"},
    },
    "required": [*DIMENSIONS, "rationales", "improvement"],
}


class LanguageModel(Protocol):
    name: str

    def recombine(
        self,
        *,
        strategy: str,
        domain_a: str,
        domain_b: str,
        text_a: str,
        text_b: str,
        temperature: float,
    ) -> dict[str, Any]: ...

    def critique(
        self,
        *,
        strategy: str,
        text_a: str,
        text_b: str,
        hypothesis: str,
        rationale: str,
        temperature: float,
    ) -> dict[str, Any]: ...


class ScriptedLanguageModel:
    """Returns queued payloads. Useful in tests."""

    name = "scripted"

    def __init__(self, hypotheses: list[dict[str, Any]], critiques: list[dict[str, Any]]) -> None:
        self._hypotheses = list(hypotheses)
        self._critiques = list(critiques)

    def recombine(self, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        if not self._hypotheses:
            raise RuntimeError("scripted recombination queue is empty")
        return self._hypotheses.pop(0)

    def critique(self, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        if not self._critiques:
            raise RuntimeError("scripted critique queue is empty")
        return self._critiques.pop(0)


class TemplateLanguageModel:
    """Offline placeholder. It juxtaposes the inputs and does not judge them."""

    name = "template"

    def recombine(
        self,
        *,
        strategy: str,
        domain_a: str,
        domain_b: str,
        text_a: str,
        text_b: str,
        temperature: float,
    ) -> dict[str, Any]:
        del temperature
        hypothesis = (
            f"[{strategy}] A conjectured link between {domain_a} ({_clip(text_a)}) "
            f"and {domain_b} ({_clip(text_b)})."
        )
        return {
            "hypothesis": hypothesis,
            "rationale": "Template model: juxtaposition only. This is not an evaluated hypothesis.",
            "no_genuine_link": False,
        }

    def critique(
        self,
        *,
        strategy: str,
        text_a: str,
        text_b: str,
        hypothesis: str,
        rationale: str,
        temperature: float,
    ) -> dict[str, Any]:
        del strategy, text_a, text_b, hypothesis, rationale, temperature
        sentence = "Template critic: no model evaluated this axis."
        return {
            "novelty": 0.5,
            "logical_coherence": 0.5,
            "falsifiability": 0.5,
            "cross_domain_utility": 0.5,
            "rationales": {name: sentence for name in DIMENSIONS},
            "improvement": "Replace the template model with a pinned evaluator before interpreting scores.",
        }


class AnthropicLanguageModel:
    """Claude via forced tool use. Temperature is passed through from the caller."""

    name = "anthropic"

    def __init__(self, config: ExperimentConfig) -> None:
        config.validate_keys(need_anthropic=True, need_voyage=False)
        self._config = config
        self._client = None

    def _client_or_create(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=self._config.anthropic_api_key)
        return self._client

    def recombine(
        self,
        *,
        strategy: str,
        domain_a: str,
        domain_b: str,
        text_a: str,
        text_b: str,
        temperature: float,
    ) -> dict[str, Any]:
        instruction = STRATEGY_INSTRUCTIONS.get(strategy, STRATEGY_INSTRUCTIONS["semantic_bridge"])
        user = (
            f"Strategy: {strategy}\n"
            f"{instruction}\n\n"
            f"Fragment A (domain {domain_a}):\n{text_a}\n\n"
            f"Fragment B (domain {domain_b}):\n{text_b}\n"
        )
        return self._tool_call(
            tool_name="submit_hypothesis",
            description="Submit the recombination hypothesis.",
            schema=_HYPOTHESIS_SCHEMA,
            system=(
                "You are a scientific hypothesis recombinator. Separate conjecture from "
                "evidence. Do not invent citations. Reply only by calling submit_hypothesis."
            ),
            user=user,
            temperature=temperature,
        )

    def critique(
        self,
        *,
        strategy: str,
        text_a: str,
        text_b: str,
        hypothesis: str,
        rationale: str,
        temperature: float,
    ) -> dict[str, Any]:
        user = (
            f"Strategy that produced the hypothesis: {strategy}\n"
            f"{CRITIC_RUBRIC}\n\n"
            f"Fragment A:\n{text_a}\n\n"
            f"Fragment B:\n{text_b}\n\n"
            f"Hypothesis:\n{hypothesis}\n\n"
            f"Author rationale:\n{rationale}\n"
        )
        return self._tool_call(
            tool_name="submit_critique",
            description="Submit the four-axis critique.",
            schema=_CRITIQUE_SCHEMA,
            system=(
                "You are a strict scientific critic in a separate pass from the author. "
                "Do not defend the hypothesis. Reply only by calling submit_critique."
            ),
            user=user,
            temperature=temperature,
        )

    def _tool_call(
        self,
        *,
        tool_name: str,
        description: str,
        schema: dict[str, Any],
        system: str,
        user: str,
        temperature: float,
    ) -> dict[str, Any]:
        message = self._client_or_create().messages.create(
            model=self._config.anthropic_model,
            max_tokens=self._config.max_tokens,
            temperature=temperature,
            system=system,
            tools=[
                {
                    "name": tool_name,
                    "description": description,
                    "input_schema": schema,
                }
            ],
            tool_choice={"type": "tool", "name": tool_name},
            messages=[{"role": "user", "content": user}],
        )
        for block in message.content:
            if getattr(block, "type", None) == "tool_use" and block.name == tool_name:
                return dict(block.input)
        texts = [
            getattr(block, "text", "")
            for block in message.content
            if getattr(block, "type", None) == "text"
        ]
        joined = "\n".join(part for part in texts if part).strip()
        if not joined:
            raise RuntimeError(f"{tool_name} returned no tool payload")
        logger.warning("%s fell back from tool use to text JSON", tool_name)
        from rem_recombinator.critic import _strip_fences
        import json

        return json.loads(_strip_fences(joined))


def build_language_model(name: str, config: ExperimentConfig) -> LanguageModel:
    if name == "template":
        return TemplateLanguageModel()
    if name == "anthropic":
        return AnthropicLanguageModel(config)
    raise ValueError(f"unknown language model {name!r}; expected 'anthropic' or 'template'")


def redact(text: str) -> str:
    """Remove common key shapes before a message is stored or logged."""
    text = re.sub(r"sk-ant-[A-Za-z0-9_\-]+", "[redacted]", text)
    text = re.sub(r"\bpa-[A-Za-z0-9_\-]{8,}", "[redacted]", text)
    return text[:500]


def _clip(text: str, limit: int = 160) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "…"
