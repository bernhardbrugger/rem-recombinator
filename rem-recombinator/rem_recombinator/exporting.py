"""Render a store export as Markdown or JSON."""

from __future__ import annotations

import json
from typing import Any


def render_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def render_markdown(payload: dict[str, Any]) -> str:
    lines = ["# REM recombination export", ""]
    runs = payload.get("runs") or []
    if not runs:
        lines.append("No experiment runs are stored.")
        lines.append("")
    for run in runs:
        hyper = run.get("hyperparameters") or {}
        lines.append(
            f"## Run {run['id']} — {run['strategy']} — seed {run['seed']}"
        )
        lines.append("")
        lines.append(
            f"Model `{hyper.get('anthropic_model', '')}`, "
            f"temperature {hyper.get('temperature')}, "
            f"critic temperature {hyper.get('critic_temperature')}, "
            f"novelty ceiling {hyper.get('novelty_ceiling')}, "
            f"incoherence floor {hyper.get('incoherence_floor')}."
        )
        lines.append("")
        candidates = run.get("candidates") or []
        if not candidates:
            lines.append("No candidates.")
            lines.append("")
            continue
        for candidate in candidates:
            critique = candidate.get("critique") or {}
            accepted = "accepted" if critique.get("accept") else candidate["filter_status"]
            score = critique.get("composite_score")
            score_text = f", composite {score:.3f}" if isinstance(score, float) else ""
            lines.append(
                f"### Candidate {candidate['id']} ({accepted}{score_text})"
            )
            lines.append("")
            parents = ", ".join(str(parent) for parent in candidate.get("parent_ids") or [])
            lines.append(f"- Strategy: `{candidate['strategy']}`")
            lines.append(f"- Parents: {parents}")
            lines.append(f"- Pair cosine: {candidate['pair_similarity']:.4f}")
            if candidate.get("novelty_score") is not None:
                nearest = candidate.get("nearest_similarity")
                nearest_text = f"{nearest:.4f}" if isinstance(nearest, (int, float)) else "n/a"
                lines.append(
                    f"- Novelty: {candidate['novelty_score']:.4f} "
                    f"(nearest fragment {candidate.get('nearest_fragment_id')}, "
                    f"cosine {nearest_text})"
                )
            lines.append("")
            lines.append(candidate.get("hypothesis") or "_No hypothesis._")
            lines.append("")
            if candidate.get("rationale"):
                lines.append(f"Rationale: {candidate['rationale']}")
                lines.append("")
            if critique:
                lines.append("| Axis | Score | Rationale |")
                lines.append("| --- | --- | --- |")
                rationales = critique.get("rationales") or {}
                for axis in (
                    "novelty",
                    "logical_coherence",
                    "falsifiability",
                    "cross_domain_utility",
                ):
                    value = critique.get(axis)
                    rendered = "" if value is None else f"{value:.3f}"
                    lines.append(f"| {axis} | {rendered} | {rationales.get(axis, '')} |")
                lines.append("")
                if critique.get("improvement"):
                    lines.append(f"Improvement: {critique['improvement']}")
                    lines.append("")
            if candidate.get("materialized_fragment_id") is not None:
                lines.append(
                    f"Stored as fragment {candidate['materialized_fragment_id']}."
                )
                lines.append("")
            for parent in candidate.get("parent_snapshot") or []:
                lines.append(
                    f"> Parent {parent.get('id')} [{parent.get('domain')}]: {parent.get('text')}"
                )
            lines.append("")
    legacy = payload.get("legacy_accepted") or []
    if legacy:
        lines.append("## Legacy accepted recombinations")
        lines.append("")
        for item in legacy:
            lines.append(f"- #{item['id']}: {item['raw_insight']}")
        lines.append("")
    return "\n".join(lines)
