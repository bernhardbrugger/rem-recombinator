"""Run one recombination experiment and persist every decision.

Selection is deterministic for corpora small enough to enumerate. The language
model and the embedder are injected, so the same control flow runs against
Claude and Voyage or against test doubles. Hyperparameters are written before
any candidate, and they never include API keys.
"""

from __future__ import annotations

import logging
import random
from dataclasses import replace
from typing import Any

from rem_recombinator.config import STRATEGIES, ExperimentConfig, load_config
from rem_recombinator.embeddings import Embedder
from rem_recombinator.llm import LanguageModel, build_language_model, redact
from rem_recombinator.models import Fragment, Hypothesis, RecombinationCandidate, RunResult
from rem_recombinator.novelty import assess_novelty
from rem_recombinator.store import Store
from rem_recombinator.critic import parse_critique
from rem_recombinator.strategies import select_pairs

logger = logging.getLogger(__name__)


def execute_run(
    *,
    store: Store,
    config: ExperimentConfig | None = None,
    embedder: Embedder,
    llm: LanguageModel,
    strategy: str | None = None,
    n: int | None = None,
    seed: int | None = None,
    materialize_accepted: bool = True,
) -> RunResult:
    """Select pairs, generate hypotheses, filter them, critique the survivors."""
    base = config or load_config()
    updates: dict[str, Any] = {}
    if strategy is not None:
        updates["strategy"] = strategy
    if n is not None:
        updates["candidates_per_run"] = n
    if seed is not None:
        updates["seed"] = seed
    active = replace(base, **updates) if updates else base
    active.validate_ranges()

    fragments = store.all_fragments()
    run_id = store.start_run(active, active.strategy)
    if len(fragments) < 2:
        logger.info("run %s skipped: fewer than two fragments", run_id)
        return RunResult(run_id, active.public_dict(), [])

    rng = random.Random(active.seed)
    quotas = _quotas(active.strategy, active.candidates_per_run)
    persisted: list[RecombinationCandidate] = []
    for name, quota in quotas:
        pairs = select_pairs(fragments, name, active, rng, quota)
        logger.info("run %s strategy %s selected %s pairs", run_id, name, len(pairs))
        for pair in pairs:
            candidate = _generate_one(
                store=store,
                config=active,
                embedder=embedder,
                llm=llm,
                fragments=fragments,
                run_id=run_id,
                pair=pair,
                materialize_accepted=materialize_accepted,
            )
            persisted.append(candidate)
            if candidate.materialized_fragment_id is not None:
                fragments = store.all_fragments()
    return RunResult(run_id, active.public_dict(), persisted)


def _generate_one(*, store, config, embedder, llm, fragments, run_id, pair, materialize_accepted):
    snapshot = [_snapshot(pair.fragment_a), _snapshot(pair.fragment_b)]
    parent_ids = [pair.fragment_a.id, pair.fragment_b.id]
    try:
        payload = llm.recombine(
            strategy=pair.strategy,
            domain_a=pair.fragment_a.domain,
            domain_b=pair.fragment_b.domain,
            text_a=pair.fragment_a.text,
            text_b=pair.fragment_b.text,
            temperature=config.temperature,
        )
        hypothesis = _hypothesis_from_payload(payload)
    except Exception as exc:
        logger.exception("recombination failed for parents %s", parent_ids)
        candidate_id = store.save_candidate(
            run_id=run_id,
            strategy=pair.strategy,
            hypothesis="",
            rationale="",
            pair_similarity=pair.similarity,
            parent_ids=parent_ids,
            parent_snapshot=snapshot,
            selection_notes=pair.selection_notes,
            filter_status="error",
            error=redact(f"{type(exc).__name__}: {exc}"),
        )
        return store.get_candidate(candidate_id)

    if hypothesis.no_genuine_link or not hypothesis.text.strip():
        candidate_id = store.save_candidate(
            run_id=run_id,
            strategy=pair.strategy,
            hypothesis=hypothesis.text.strip(),
            rationale=hypothesis.rationale,
            pair_similarity=pair.similarity,
            parent_ids=parent_ids,
            parent_snapshot=snapshot,
            selection_notes=pair.selection_notes,
            filter_status="no_genuine_link",
        )
        return store.get_candidate(candidate_id)

    try:
        vector = embedder.embed_text(hypothesis.text, input_type="document")
    except Exception as exc:
        logger.exception("hypothesis embedding failed")
        candidate_id = store.save_candidate(
            run_id=run_id,
            strategy=pair.strategy,
            hypothesis=hypothesis.text,
            rationale=hypothesis.rationale,
            pair_similarity=pair.similarity,
            parent_ids=parent_ids,
            parent_snapshot=snapshot,
            selection_notes=pair.selection_notes,
            filter_status="error",
            error=redact(f"{type(exc).__name__}: {exc}"),
        )
        return store.get_candidate(candidate_id)

    decision = assess_novelty(
        vector,
        fragments,
        ceiling=config.novelty_ceiling,
        floor=config.incoherence_floor,
    )
    candidate_id = store.save_candidate(
        run_id=run_id,
        strategy=pair.strategy,
        hypothesis=hypothesis.text,
        rationale=hypothesis.rationale,
        pair_similarity=pair.similarity,
        parent_ids=parent_ids,
        parent_snapshot=snapshot,
        selection_notes=pair.selection_notes,
        filter_status=decision.status,
        novelty_score=decision.novelty_score,
        nearest_fragment_id=decision.nearest_fragment_id,
        nearest_similarity=decision.nearest_similarity,
    )
    if decision.status != "kept":
        return store.get_candidate(candidate_id)

    try:
        critique_payload = llm.critique(
            strategy=pair.strategy,
            text_a=pair.fragment_a.text,
            text_b=pair.fragment_b.text,
            hypothesis=hypothesis.text,
            rationale=hypothesis.rationale,
            temperature=config.critic_temperature,
        )
        report = parse_critique(critique_payload, config)
    except Exception as exc:
        logger.exception("critique failed for candidate %s", candidate_id)
        report = parse_critique({}, config)
        report.parse_error = redact(f"{type(exc).__name__}: {exc}")
    store.save_critique(candidate_id, report)
    saved = store.get_candidate(candidate_id)
    if materialize_accepted and report.accept:
        store.materialize_candidate(candidate_id, vector)
        saved = store.get_candidate(candidate_id)
    return saved


def _hypothesis_from_payload(payload: dict[str, Any]) -> Hypothesis:
    if not isinstance(payload, dict):
        raise ValueError("recombination payload must be an object")
    text = payload.get("hypothesis", "")
    rationale = payload.get("rationale", "")
    flag = payload.get("no_genuine_link", False)
    if not isinstance(text, str) or not isinstance(rationale, str) or not isinstance(flag, bool):
        raise ValueError("recombination payload fields have the wrong types")
    return Hypothesis(text=text.strip(), rationale=rationale.strip(), no_genuine_link=flag)


def _snapshot(fragment: Fragment) -> dict[str, Any]:
    return {
        "id": fragment.id,
        "source": fragment.source,
        "domain": fragment.domain,
        "text": fragment.text,
        "created_at": fragment.created_at,
    }


def _quotas(strategy: str, n: int) -> list[tuple[str, int]]:
    names = list(STRATEGIES) if strategy == "all" else [strategy]
    base, extra = divmod(n, len(names))
    quotas = []
    for index, name in enumerate(names):
        count = base + (1 if index < extra else 0)
        if count:
            quotas.append((name, count))
    return quotas
