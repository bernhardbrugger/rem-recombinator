"""Threshold reports stay descriptive: they rank bands and leave defaults alone."""

import json

import pytest

from rem_recombinator.cli import main
from rem_recombinator.evaluation import (
    BANNER,
    CritiqueLabel,
    HoldoutCase,
    NoveltyLabel,
    PairLabel,
    best_operating_point,
    critique_agreement,
    midpoint_holdout,
    render_report,
    sweep_novelty,
    sweep_strategy,
)
from rem_recombinator.models import CritiqueReport
from tests.fakes import unit


def _add(store, text, domain, vector):
    return store.add_fragment(text, vector, source="test", domain=domain)


def test_bridge_sweep_can_prefer_a_wider_band_without_editing_defaults(store, config):
    positive_b = unit(0.40, (1 - 0.40**2) ** 0.5, 0)
    _add(store, "alpha claim", "physics", unit(1, 0, 0))
    _add(store, "beta claim", "biology", positive_b)
    _add(store, "orthogonal claim", "chemistry", unit(0, 0, 1))
    fragments = store.all_fragments()
    labels = [
        PairLabel(1, 2, True),
        PairLabel(1, 3, False),
    ]
    points = sweep_strategy(fragments, labels, config, "semantic_bridge")
    best = best_operating_point(
        points,
        default_low=config.min_similarity_for_candidate,
        default_high=config.max_similarity_for_candidate,
    )
    assert best is not None
    assert best.f1 == pytest.approx(1.0)
    assert best.low == pytest.approx(0.05)
    assert best.high == pytest.approx(0.45)
    assert config.max_similarity_for_candidate == pytest.approx(0.35)


def test_novelty_sweep_keeps_the_default_when_it_is_perfect():
    labels = [
        NoveltyLabel(0.95, "too_similar"),
        NoveltyLabel(0.50, "kept"),
        NoveltyLabel(0.01, "incoherent"),
    ]
    points = sweep_novelty(labels)
    best = best_operating_point(points, default_low=0.05, default_high=0.88)
    assert best is not None
    assert best.f1 == pytest.approx(1.0)
    assert best.low == pytest.approx(0.05)
    assert best.high == pytest.approx(0.88)
    ceiling_99 = [point for point in points if point.high == pytest.approx(0.95) and point.low == pytest.approx(0.05)]
    assert ceiling_99[0].f1 < 1.0


def test_holdout_and_critic_agreement(store):
    _add(store, "left", "physics", unit(1, 0, 0))
    _add(store, "right", "biology", unit(0, 1, 0))
    _add(store, "the midpoint", "bridge", unit(1, 1, 0))
    _add(store, "distractor", "other", unit(0, 0, 1))
    hits, total = midpoint_holdout(
        store.all_fragments(),
        [HoldoutCase(1, 2, 3)],
    )
    assert (hits, total) == (1, 1)
    misses, _ = midpoint_holdout(store.all_fragments(), [HoldoutCase(1, 2, 4)])
    assert misses == 0

    agreement = critique_agreement(
        [
            CritiqueLabel(
                7,
                {
                    "novelty": 0.8,
                    "logical_coherence": 0.8,
                    "falsifiability": 0.8,
                    "cross_domain_utility": 0.8,
                },
                True,
            )
        ],
        {
            7: {
                "novelty": 0.6,
                "logical_coherence": 0.6,
                "falsifiability": 0.6,
                "cross_domain_utility": 0.6,
                "accept": True,
            }
        },
    )
    assert agreement["novelty"] == pytest.approx(0.2)
    assert agreement["accept_agreement"] == pytest.approx(1.0)


def test_report_and_cli_leave_a_banner(store, config, tmp_path, capsys):
    _add(store, "alpha claim", "physics", unit(1, 0, 0))
    _add(store, "beta claim", "biology", unit(0.40, (1 - 0.40**2) ** 0.5, 0))
    _add(store, "orthogonal claim", "chemistry", unit(0, 0, 1))
    from rem_recombinator.evaluation import LabelSet

    report = render_report(
        store,
        config,
        LabelSet(
            bridge=[PairLabel(1, 2, True), PairLabel(1, 3, False)],
            analogy=[],
            dialectic=[],
            novelty=[NoveltyLabel(0.95, "too_similar"), NoveltyLabel(0.50, "kept")],
            critiques=[],
            holdout=[],
            note="fixture",
        ),
    )
    assert report.startswith(BANNER)
    assert "Best F1 1.000 at [0.05, 0.45]" in report
    assert config.max_similarity_for_candidate == pytest.approx(0.35)

    path = tmp_path / "labels.json"
    path.write_text(
        json.dumps(
            {
                "bridge": [
                    {"left_id": 1, "right_id": 2, "positive": True},
                    {"left_id": 1, "right_id": 3, "positive": False},
                ]
            }
        ),
        encoding="utf-8",
    )
    code = main(["--db", str(store.db_path), "evaluate", str(path)])
    assert code == 0
    output = capsys.readouterr().out
    assert "Defaults unchanged" in output
    assert "0.45" in output


def test_missing_label_id_is_an_error(store, config):
    _add(store, "only", "physics", unit(1, 0, 0))
    with pytest.raises(KeyError, match="label ids"):
        sweep_strategy(store.all_fragments(), [PairLabel(1, 9, True)], config, "semantic_bridge")


def test_empty_label_file_prints_banner(tmp_path, capsys):
    database = tmp_path / "empty.db"
    labels = tmp_path / "labels.json"
    labels.write_text('{"note": "schema only"}', encoding="utf-8")
    code = main(["--db", str(database), "evaluate", str(labels)])
    assert code == 0
    output = capsys.readouterr().out
    assert "Defaults unchanged" in output
    assert "No labeled sections." in output


def test_stored_critique_round_trip_is_not_required_for_mae_types():
    """The agreement helper accepts the same fields a CritiqueReport stores."""
    report = CritiqueReport(
        novelty=0.5,
        logical_coherence=0.5,
        falsifiability=0.5,
        cross_domain_utility=0.5,
        rationales={name: "x" for name in ("novelty", "logical_coherence", "falsifiability", "cross_domain_utility")},
        improvement="y",
        composite_score=0.5,
        accept=False,
    )
    assert report.accept is False
