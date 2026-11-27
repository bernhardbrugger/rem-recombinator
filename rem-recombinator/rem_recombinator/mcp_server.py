"""STDIO MCP server.

Nothing in this module writes to stdout. The MCP transport owns stdout, and
logs are attached to stderr only when ``serve`` runs.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import replace
from typing import Any

from rem_recombinator.config import STRATEGY_CHOICES, ExperimentConfig, load_config
from rem_recombinator.embeddings import Embedder, build_embedder
from rem_recombinator.llm import LanguageModel, build_language_model
from rem_recombinator.recombinator import execute_run
from rem_recombinator.store import Store

logger = logging.getLogger(__name__)

_RUN_FIELDS = {"strategy", "candidates", "seed", "materialize"}
_LIST_FIELDS = {"limit"}
_LINEAGE_FIELDS = {"fragment_id"}


def tool_specs() -> list[dict[str, Any]]:
    """JSON tool descriptors. Kept free of the MCP SDK so they can be tested offline."""
    return [
        {
            "name": "run_rem_cycle",
            "description": (
                "Run one REM recombination experiment on the local fragment store. "
                "Selects pairs with the requested strategy (semantic_bridge, "
                "analogical_transfer, dialectic_synthesis, or all), asks the "
                "configured model for a hypothesis, drops hypotheses whose embedding "
                "is closer than the novelty ceiling (default cosine 0.88) or farther "
                "than the incoherence floor, then scores survivors on novelty, "
                "logical coherence, falsifiability, and cross-domain utility. "
                "Accepted hypotheses are stored as fragments with parent lineage. "
                "Returns the run id, filter status, scores, and hypothesis text."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "strategy": {
                        "type": "string",
                        "enum": list(STRATEGY_CHOICES),
                        "description": "Selection strategy. Defaults to semantic_bridge.",
                    },
                    "candidates": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "description": "How many pairs to request. Defaults to 5.",
                    },
                    "seed": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "RNG seed used only when the corpus is too large to enumerate.",
                    },
                    "materialize": {
                        "type": "boolean",
                        "description": "Store accepted hypotheses as fragments. Defaults to true.",
                    },
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "list_accepted_insights",
            "description": (
                "List hypotheses the critic accepted, newest first, including parent ids, "
                "strategy, composite score, and the four axis scores."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "description": "Maximum rows. Defaults to 20.",
                    }
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "store_stats",
            "description": (
                "Return fragment counts by domain and source, run and candidate counts, "
                "filter-status counts, and mean critic scores."
            ),
            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        },
        {
            "name": "fragment_lineage",
            "description": (
                "Return one fragment, the parents it was recombined from, the child "
                "fragments derived from it, and the candidate ids that used it as a parent."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "fragment_id": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Primary key of the fragment.",
                    }
                },
                "required": ["fragment_id"],
                "additionalProperties": False,
            },
        },
    ]


def validate_arguments(name: str, arguments: dict[str, Any] | None) -> dict[str, Any]:
    """Check tool arguments. Raises ValueError with a message a client can show."""
    raw = dict(arguments or {})
    if name == "run_rem_cycle":
        _reject_unknown(raw, _RUN_FIELDS)
        strategy = raw.get("strategy", "semantic_bridge")
        if strategy not in STRATEGY_CHOICES:
            raise ValueError(
                "strategy must be one of " + ", ".join(STRATEGY_CHOICES)
            )
        candidates = raw.get("candidates", 5)
        _require_int(candidates, "candidates", minimum=1, maximum=20)
        seed = raw.get("seed")
        if seed is not None:
            _require_int(seed, "seed", minimum=0, maximum=None)
        materialize = raw.get("materialize", True)
        if not isinstance(materialize, bool):
            raise ValueError("materialize must be a boolean")
        return {
            "strategy": strategy,
            "candidates": candidates,
            "seed": seed,
            "materialize": materialize,
        }
    if name == "list_accepted_insights":
        _reject_unknown(raw, _LIST_FIELDS)
        limit = raw.get("limit", 20)
        _require_int(limit, "limit", minimum=1, maximum=100)
        return {"limit": limit}
    if name == "store_stats":
        _reject_unknown(raw, set())
        return {}
    if name == "fragment_lineage":
        _reject_unknown(raw, _LINEAGE_FIELDS)
        if "fragment_id" not in raw:
            raise ValueError("fragment_id is required")
        _require_int(raw["fragment_id"], "fragment_id", minimum=1, maximum=None)
        return {"fragment_id": raw["fragment_id"]}
    raise ValueError(f"unknown tool: {name}")


def dispatch(
    name: str,
    arguments: dict[str, Any] | None,
    *,
    store: Store,
    config: ExperimentConfig,
    embedder: Embedder | None = None,
    llm: LanguageModel | None = None,
) -> str:
    """Run one tool and return text. Does not write to stdout."""
    try:
        clean = validate_arguments(name, arguments)
    except ValueError as exc:
        return f"Invalid arguments: {exc}"
    if name == "store_stats":
        return _format_stats(store.stats())
    if name == "fragment_lineage":
        try:
            payload = store.fragment_lineage(clean["fragment_id"])
        except KeyError:
            return f"No fragment {clean['fragment_id']}."
        return _format_lineage(payload)
    if name == "list_accepted_insights":
        rows = store.accepted_candidates(limit=clean["limit"])
        if not rows:
            return "No accepted insights stored yet."
        return "\n".join(_format_candidate(row) for row in rows)
    if name == "run_rem_cycle":
        active_embedder = embedder or build_embedder("voyage", config)
        active_llm = llm or build_language_model("anthropic", config)
        active = replace(
            config,
            strategy=clean["strategy"],
            candidates_per_run=clean["candidates"],
            seed=config.seed if clean["seed"] is None else clean["seed"],
        )
        result = execute_run(
            store=store,
            config=active,
            embedder=active_embedder,
            llm=active_llm,
            materialize_accepted=clean["materialize"],
        )
        if not result.candidates:
            return (
                f"Run {result.run_id}: not enough fragments, or no pair passed selection. "
                f"Fragments in store: {store.count_fragments()}."
            )
        lines = [
            f"Run {result.run_id}: {len(result.candidates)} candidates, "
            f"{len(result.accepted)} accepted."
        ]
        lines.extend(_format_candidate(candidate) for candidate in result.candidates)
        return "\n".join(lines)
    raise ValueError(f"unknown tool: {name}")


def serve() -> None:
    """Start the STDIO server. Logging is attached to stderr inside this function."""
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    import asyncio

    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import TextContent, Tool

    app = Server("rem-recombinator")
    config = load_config()
    store = Store(config.db_path)

    @app.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(name=spec["name"], description=spec["description"], inputSchema=spec["inputSchema"])
            for spec in tool_specs()
        ]

    @app.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        logger.info("tool %s", name)
        text = dispatch(name, arguments, store=store, config=config)
        return [TextContent(type="text", text=text)]

    async def runner() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await app.run(read_stream, write_stream, app.create_initialization_options())

    try:
        asyncio.run(runner())
    finally:
        store.close()


def _require_int(value: Any, name: str, *, minimum: int, maximum: int | None) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        upper = "" if maximum is None else f" and <= {maximum}"
        raise ValueError(f"{name} must be >= {minimum}{upper}")


def _reject_unknown(arguments: dict[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(arguments) - allowed)
    if unknown:
        raise ValueError("unknown arguments: " + ", ".join(unknown))


def _format_stats(stats: dict[str, Any]) -> str:
    lines = [f"Fragments: {stats['fragments']}"]
    for domain, count in stats["by_domain"].items():
        lines.append(f"domain {domain}: {count}")
    lines.append(f"Runs: {stats['runs']}")
    lines.append(f"Candidates: {stats['candidates']}")
    lines.append(f"Accepted: {stats['accepted']}")
    for status, count in stats["by_filter_status"].items():
        lines.append(f"{status}: {count}")
    return "\n".join(lines)


def _format_lineage(payload: dict[str, Any]) -> str:
    fragment = payload["fragment"]
    parents = ", ".join(str(item["id"]) for item in payload["parents"]) or "none"
    children = ", ".join(str(item["id"]) for item in payload["children"]) or "none"
    candidates = ", ".join(str(item) for item in payload["candidate_ids"]) or "none"
    return (
        f"Fragment {fragment['id']} [{fragment['domain']}] {fragment['text']}\n"
        f"Parents: {parents}\n"
        f"Children: {children}\n"
        f"Candidates: {candidates}"
    )


def _format_candidate(candidate: Any) -> str:
    critique = candidate.critique
    if critique is None:
        score = "n/a"
    else:
        score = (
            f"composite {critique.composite_score:.3f} "
            f"(novelty {critique.novelty:.2f}, coherence {critique.logical_coherence:.2f}, "
            f"falsifiability {critique.falsifiability:.2f}, utility {critique.cross_domain_utility:.2f}, "
            f"accept {critique.accept})"
        )
    hypothesis = candidate.hypothesis or "(none)"
    return (
        f"#{candidate.id} {candidate.strategy} {candidate.filter_status} "
        f"parents={candidate.parent_ids} {score}\n  {hypothesis}"
    )


if __name__ == "__main__":
    serve()
