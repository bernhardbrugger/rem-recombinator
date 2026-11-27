"""Critic parsing and an offline recombination cycle."""

from pathlib import Path

import pytest

from rem_recombinator.critic import DIMENSIONS, parse_critique
from rem_recombinator.llm import ScriptedLanguageModel, redact
from rem_recombinator.recombinator import execute_run
from tests.fakes import MapEmbedder, unit


def _critique(**overrides):
    payload = {
        "novelty": 0.8,
        "logical_coherence": 0.8,
        "falsifiability": 0.8,
        "cross_domain_utility": 0.8,
        "rationales": {name: f"Reason for {name}." for name in DIMENSIONS},
        "improvement": "Name the measurement that would refute it.",
    }
    payload.update(overrides)
    return payload


def test_parse_critique_accepts_a_complete_payload(config):
    report = parse_critique(_critique(), config)
    assert report.accept
    assert report.composite_score == pytest.approx(0.8)
    assert report.parse_error is None


def test_parse_critique_strips_fences_and_rejects_a_weak_axis(config):
    fenced = """```json
    {"novelty": 1, "logical_coherence": 1, "falsifiability": 1, "cross_domain_utility": 0.2,
     "rationales": {"novelty": "n", "logical_coherence": "c", "falsifiability": "f", "cross_domain_utility": "u"},
     "improvement": "Raise utility."}
    ```"""
    report = parse_critique(fenced, config)
    assert report.accept is False
    assert report.composite_score == pytest.approx(0.84)


def test_parse_critique_rejects_out_of_range_and_garbage(config):
    broken = _critique(novelty=1.2)
    assert parse_critique(broken, config).accept is False
    assert parse_critique(broken, config).parse_error
    assert parse_critique("not json", config).parse_error
    assert parse_critique({"novelty": True}, config).accept is False


def _parents(store):
    left = "Closed systems spread heat."
    right = "Crystal lattices tile space."
    store.add_fragment(left, unit(1, 0, 0), source="test", domain="physics")
    store.add_fragment(right, unit(0.25, (1 - 0.25**2) ** 0.5, 0), source="test", domain="physics")
    return left, right


def test_cycle_accepts_a_mid_range_hypothesis_and_records_lineage(store, config):
    left, right = _parents(store)
    hypothesis = "Heat flow and lattice tiling share a conserved bookkeeping step."
    embedder = MapEmbedder(
        {
            left: unit(1, 0, 0),
            right: unit(0.25, (1 - 0.25**2) ** 0.5, 0),
            hypothesis: unit(0.2, 0.2, 0.96),
        }
    )
    model = ScriptedLanguageModel(
        hypotheses=[{"hypothesis": hypothesis, "rationale": "Shared bookkeeping.", "no_genuine_link": False}],
        critiques=[_critique()],
    )
    result = execute_run(
        store=store,
        config=config,
        embedder=embedder,
        llm=model,
        strategy="semantic_bridge",
        n=1,
        seed=0,
    )
    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.filter_status == "kept"
    assert candidate.critique is not None and candidate.critique.accept
    assert candidate.materialized_fragment_id is not None
    child = store.get_fragment(candidate.materialized_fragment_id)
    lineage = store.fragment_lineage(child.id)
    assert len(lineage["parents"]) == 2
    stored = store.runs()[0]["hyperparameters"]
    assert stored["seed"] == 0
    assert stored["strategy"] == "semantic_bridge"
    assert "anthropic_api_key" not in stored
    assert result.hyperparameters["novelty_ceiling"] == 0.88


def test_cycle_drops_paraphrases_without_calling_the_critic(store, config):
    left, right = _parents(store)
    hypothesis = "Closed systems spread heat."
    parent = unit(1, 0, 0)
    embedder = MapEmbedder(
        {
            left: parent,
            right: unit(0.25, (1 - 0.25**2) ** 0.5, 0),
            hypothesis: parent,
        }
    )
    model = ScriptedLanguageModel(
        hypotheses=[{"hypothesis": hypothesis, "rationale": "restatement", "no_genuine_link": False}],
        critiques=[_critique()],
    )
    result = execute_run(
        store=store, config=config, embedder=embedder, llm=model, strategy="semantic_bridge", n=1
    )
    assert result.candidates[0].filter_status == "too_similar"
    assert result.candidates[0].critique is None
    assert model._critiques  # critic was not called


def test_cycle_drops_incoherent_vectors_and_redacts_errors(store, config):
    left, right = _parents(store)
    hypothesis = "An orthogonal conjecture."
    embedder = MapEmbedder(
        {
            left: unit(1, 0, 0),
            right: unit(0.25, (1 - 0.25**2) ** 0.5, 0),
            hypothesis: unit(0, 0, 1),
        }
    )
    model = ScriptedLanguageModel(
        hypotheses=[{"hypothesis": hypothesis, "rationale": "none", "no_genuine_link": False}],
        critiques=[],
    )
    result = execute_run(
        store=store, config=config, embedder=embedder, llm=model, strategy="semantic_bridge", n=1
    )
    assert result.candidates[0].filter_status == "incoherent"

    class Boom:
        name = "boom"

        def recombine(self, **kwargs):
            del kwargs
            raise RuntimeError("provider rejected sk-ant-secretvalue")

        def critique(self, **kwargs):
            raise AssertionError("critique should not run")

    failed = execute_run(
        store=store, config=config, embedder=embedder, llm=Boom(), strategy="semantic_bridge", n=1
    )
    assert failed.candidates[0].filter_status == "error"
    assert "sk-ant-" not in (failed.candidates[0].error or "")
    assert "[redacted]" in (failed.candidates[0].error or "")
    assert "secretvalue" not in redact("sk-ant-secretvalue")


def test_no_genuine_link_is_stored_without_a_critique(store, config):
    left, right = _parents(store)
    embedder = MapEmbedder({left: unit(1, 0, 0), right: unit(0, 1, 0)})
    model = ScriptedLanguageModel(
        hypotheses=[{"hypothesis": "", "rationale": "No mechanism.", "no_genuine_link": True}],
        critiques=[],
    )
    result = execute_run(
        store=store, config=config, embedder=embedder, llm=model, strategy="semantic_bridge", n=1
    )
    assert result.candidates[0].filter_status == "no_genuine_link"
    assert result.candidates[0].critique is None


def test_library_modules_do_not_print():
    root = Path(__file__).resolve().parents[1] / "rem_recombinator"
    offenders = []
    for path in root.glob("*.py"):
        if path.name == "cli.py":
            continue
        if "print(" in path.read_text(encoding="utf-8"):
            offenders.append(path.name)
    assert offenders == []
