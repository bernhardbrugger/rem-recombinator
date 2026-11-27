"""Threshold reports against a labeled set.

The sweep measures precision and recall of the current selectors. It does not
write a new default. A high F1 on one reader's labels is a result to publish
beside the labels, not a reason for this module to edit ``REM_MAX_SIM``.

Pair metrics use only labeled pairs. An unlabeled pair that the selector
returns is ignored, so a partial label set does not punish the model for
pairs nobody judged. A labeled positive that is missing from the selection is
a false negative. A labeled negative that is selected is a false positive.

Bridge metrics include the cluster constraint, not only the cosine band.
Dialectic and analogy metrics include their other eligibility rules the same
way. The sweep asks whether a pair is in the eligible set, so the ranker's
target similarity does not affect the score; it is moved inside the band only
to keep the config valid.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from rem_recombinator.config import ExperimentConfig
from rem_recombinator.critic import DIMENSIONS
from rem_recombinator.embeddings import cosine_similarity
from rem_recombinator.models import Fragment
from rem_recombinator.novelty import status_from_nearest
from rem_recombinator.store import Store
from rem_recombinator.strategies import midpoint, select_pairs

BANNER = "Defaults unchanged. This report does not refit REM_* thresholds."

_STRATEGY_BANDS: dict[str, tuple[str, str]] = {
    "semantic_bridge": ("min_similarity_for_candidate", "max_similarity_for_candidate"),
    "analogical_transfer": ("analogical_min_similarity", "analogical_max_similarity"),
    "dialectic_synthesis": ("dialectic_min_similarity", "dialectic_max_similarity"),
}

_SECTION_STRATEGY = {
    "bridge": "semantic_bridge",
    "analogy": "analogical_transfer",
    "dialectic": "dialectic_synthesis",
}

DEFAULT_LOWS = (0.0, 0.05, 0.10, 0.15)
DEFAULT_HIGHS = (0.25, 0.35, 0.45, 0.55, 0.75)
DEFAULT_FLOORS = (0.0, 0.05, 0.10, 0.20)
DEFAULT_CEILINGS = (0.70, 0.80, 0.88, 0.95)


@dataclass(frozen=True)
class PairLabel:
    left_id: int
    right_id: int
    positive: bool


@dataclass(frozen=True)
class NoveltyLabel:
    nearest_similarity: float
    gold: str


@dataclass(frozen=True)
class CritiqueLabel:
    candidate_id: int
    scores: dict[str, float]
    accept: bool | None


@dataclass(frozen=True)
class HoldoutCase:
    left_id: int
    right_id: int
    held_out_id: int


@dataclass(frozen=True)
class OperatingPoint:
    low: float
    high: float
    precision: float | None
    recall: float | None
    f1: float | None
    true_positive: int
    false_positive: int
    false_negative: int


@dataclass
class LabelSet:
    bridge: list[PairLabel]
    analogy: list[PairLabel]
    dialectic: list[PairLabel]
    novelty: list[NoveltyLabel]
    critiques: list[CritiqueLabel]
    holdout: list[HoldoutCase]
    note: str = ""


def load_label_file(path: Path) -> LabelSet:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("label file must be a JSON object")
    return LabelSet(
        bridge=_pair_labels(data.get("bridge") or [], "bridge"),
        analogy=_pair_labels(data.get("analogy") or [], "analogy"),
        dialectic=_pair_labels(data.get("dialectic") or [], "dialectic"),
        novelty=_novelty_labels(data.get("novelty") or []),
        critiques=_critique_labels(data.get("critiques") or []),
        holdout=_holdout_cases(data.get("holdout") or []),
        note=str(data.get("note") or ""),
    )


def sweep_strategy(
    fragments: list[Fragment],
    labels: list[PairLabel],
    config: ExperimentConfig,
    strategy: str,
    *,
    lows: tuple[float, ...] = DEFAULT_LOWS,
    highs: tuple[float, ...] = DEFAULT_HIGHS,
) -> list[OperatingPoint]:
    """Precision and recall of one strategy across a band grid."""
    if strategy not in _STRATEGY_BANDS:
        raise ValueError(f"unknown strategy {strategy!r}")
    _require_ids(fragments, _ids_in_pairs(labels))
    low_name, high_name = _STRATEGY_BANDS[strategy]
    points: list[OperatingPoint] = []
    for low in lows:
        for high in highs:
            if not 0.0 <= low < high <= 1.0:
                continue
            updates: dict[str, float] = {low_name: low, high_name: high}
            if strategy == "dialectic_synthesis":
                updates["dialectic_target_similarity"] = (low + high) / 2
            try:
                active = replace(config, **updates)
                active.validate_ranges()
            except ValueError:
                continue
            selected = select_pairs(
                fragments,
                strategy,
                active,
                random.Random(config.seed),
                _all_pairs(len(fragments)),
            )
            predicted = {pair_key(item.fragment_a.id, item.fragment_b.id) for item in selected}
            points.append(_point(low, high, predicted, labels))
    return points


def sweep_novelty(
    labels: list[NoveltyLabel],
    *,
    floors: tuple[float, ...] = DEFAULT_FLOORS,
    ceilings: tuple[float, ...] = DEFAULT_CEILINGS,
) -> list[OperatingPoint]:
    """Exact-status accuracy is reported as precision=recall=accuracy on the gold set.

    The positive class is a correct status, so precision and recall are equal
    to accuracy. That keeps one table shape for every sweep.
    """
    points: list[OperatingPoint] = []
    allowed = {"kept", "too_similar", "incoherent"}
    for label in labels:
        if label.gold not in allowed:
            raise ValueError(f"novelty gold must be one of {sorted(allowed)}")
    for floor in floors:
        for ceiling in ceilings:
            if not 0.0 <= floor < ceiling <= 1.0:
                continue
            correct = 0
            for label in labels:
                predicted = status_from_nearest(
                    label.nearest_similarity, ceiling=ceiling, floor=floor
                )
                if predicted == label.gold:
                    correct += 1
            wrong = len(labels) - correct
            accuracy = (correct / len(labels)) if labels else None
            points.append(
                OperatingPoint(
                    low=floor,
                    high=ceiling,
                    precision=accuracy,
                    recall=accuracy,
                    f1=accuracy,
                    true_positive=correct,
                    false_positive=wrong,
                    false_negative=wrong,
                )
            )
    return points


def best_operating_point(
    points: list[OperatingPoint],
    *,
    default_low: float,
    default_high: float,
) -> OperatingPoint | None:
    """Highest F1. Ties prefer the band closest to the current defaults."""
    scored = [point for point in points if point.f1 is not None]
    if not scored:
        return None
    return min(
        scored,
        key=lambda point: (
            -point.f1,
            -(point.recall or 0.0),
            abs(point.low - default_low) + abs(point.high - default_high),
            point.high,
            -point.low,
        ),
    )


def critique_agreement(
    gold: list[CritiqueLabel],
    predicted: dict[int, dict[str, Any]],
) -> dict[str, float | None]:
    """Mean absolute error per axis, plus agreement on the accept bit."""
    if not gold:
        return {}
    missing = [item.candidate_id for item in gold if item.candidate_id not in predicted]
    if missing:
        raise KeyError(f"no stored critique for candidate ids {missing}")
    totals = {axis: 0.0 for axis in DIMENSIONS}
    accept_hits = 0
    accept_n = 0
    for item in gold:
        row = predicted[item.candidate_id]
        for axis in DIMENSIONS:
            totals[axis] += abs(item.scores[axis] - float(row[axis]))
        if item.accept is not None:
            accept_n += 1
            if bool(row["accept"]) is item.accept:
                accept_hits += 1
    result: dict[str, float | None] = {
        axis: totals[axis] / len(gold) for axis in DIMENSIONS
    }
    result["accept_agreement"] = (accept_hits / accept_n) if accept_n else None
    return result


def midpoint_holdout(fragments: list[Fragment], cases: list[HoldoutCase]) -> tuple[int, int]:
    """How often the held-out fragment is the nearest neighbour of the pair midpoint.

    The two parents are excluded. Ties break toward the smaller id.
    """
    by_id = {fragment.id: fragment for fragment in fragments}
    needed = [case.left_id for case in cases]
    needed += [case.right_id for case in cases]
    needed += [case.held_out_id for case in cases]
    missing = sorted({item for item in needed if item not in by_id})
    if missing:
        raise KeyError(f"hold-out ids not in the store: {missing}")
    hits = 0
    for case in cases:
        left = by_id[case.left_id]
        right = by_id[case.right_id]
        point = midpoint(left.embedding, right.embedding)
        others = [fragment for fragment in fragments if fragment.id not in (left.id, right.id)]
        if not others:
            continue
        nearest = max(
            others,
            key=lambda fragment: (cosine_similarity(point, fragment.embedding), -(fragment.id or 0)),
        )
        if nearest.id == case.held_out_id:
            hits += 1
    return hits, len(cases)


def render_report(store: Store, config: ExperimentConfig, labels: LabelSet) -> str:
    """Text report. The config object is read and not replaced in the caller."""
    fragments = store.all_fragments()
    lines = [BANNER]
    if labels.note:
        lines.append(labels.note)
    total_pairs = len(fragments) * (len(fragments) - 1) // 2
    if total_pairs > config.pair_enumeration_cap:
        lines.append(
            f"Corpus has {total_pairs} pairs, above REM_PAIR_CAP "
            f"({config.pair_enumeration_cap}). Strategy metrics are on a seeded sample."
        )
    sections = (
        ("bridge", labels.bridge, config.min_similarity_for_candidate, config.max_similarity_for_candidate),
        ("analogy", labels.analogy, config.analogical_min_similarity, config.analogical_max_similarity),
        ("dialectic", labels.dialectic, config.dialectic_min_similarity, config.dialectic_max_similarity),
    )
    wrote = False
    for name, group, default_low, default_high in sections:
        if not group:
            continue
        wrote = True
        points = sweep_strategy(fragments, group, config, _SECTION_STRATEGY[name])
        best = best_operating_point(points, default_low=default_low, default_high=default_high)
        lines.append("")
        lines.append(
            f"## {name}  default band [{default_low:.2f}, {default_high:.2f}]  "
            f"labeled {len(group)}"
        )
        if name == "bridge":
            lines.append("Eligibility includes the cluster split, not only the cosine band.")
        if best is None:
            lines.append("No operating point had a defined F1.")
            continue
        lines.append(
            f"Best F1 { _fmt(best.f1) } at [{best.low:.2f}, {best.high:.2f}] "
            f"(precision {_fmt(best.precision)}, recall {_fmt(best.recall)}, "
            f"tp {best.true_positive}, fp {best.false_positive}, fn {best.false_negative})."
        )
    if labels.novelty:
        wrote = True
        points = sweep_novelty(labels.novelty)
        best = best_operating_point(
            points,
            default_low=config.incoherence_floor,
            default_high=config.novelty_ceiling,
        )
        lines.append("")
        lines.append(
            f"## novelty  default floor {config.incoherence_floor:.2f} "
            f"ceiling {config.novelty_ceiling:.2f}  labeled {len(labels.novelty)}"
        )
        if best is not None:
            lines.append(
                f"Best status accuracy { _fmt(best.f1) } at floor {best.low:.2f}, "
                f"ceiling {best.high:.2f}."
            )
    if labels.critiques:
        wrote = True
        predicted = _stored_critiques(store, [item.candidate_id for item in labels.critiques])
        agreement = critique_agreement(labels.critiques, predicted)
        lines.append("")
        lines.append(f"## critic agreement  n {len(labels.critiques)}")
        for axis in DIMENSIONS:
            lines.append(f"  {axis} MAE {agreement[axis]:.3f}")
        if agreement["accept_agreement"] is not None:
            lines.append(f"  accept agreement {agreement['accept_agreement']:.3f}")
    if labels.holdout:
        wrote = True
        hits, total = midpoint_holdout(fragments, labels.holdout)
        lines.append("")
        lines.append(
            f"## midpoint hold-out  {hits}/{total} "
            f"({(hits / total) if total else 0:.3f})"
        )
        lines.append(
            "A hit means the held-out fragment is the nearest neighbour of the parent midpoint."
        )
    if not wrote:
        lines.append("No labeled sections.")
    lines.append("")
    return "\n".join(lines)


def pair_key(left_id: int | None, right_id: int | None) -> tuple[int, int]:
    if left_id is None or right_id is None:
        raise ValueError("pair ids are required")
    return (left_id, right_id) if left_id <= right_id else (right_id, left_id)


def _point(
    low: float,
    high: float,
    predicted: set[tuple[int, int]],
    labels: list[PairLabel],
) -> OperatingPoint:
    true_positive = false_positive = false_negative = 0
    for label in labels:
        hit = pair_key(label.left_id, label.right_id) in predicted
        if label.positive and hit:
            true_positive += 1
        elif label.positive and not hit:
            false_negative += 1
        elif not label.positive and hit:
            false_positive += 1
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    f1 = _f1(precision, recall)
    return OperatingPoint(
        low=low,
        high=high,
        precision=precision,
        recall=recall,
        f1=f1,
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
    )


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None and recall is None:
        return None
    precision_value = 0.0 if precision is None else precision
    recall_value = 0.0 if recall is None else recall
    if precision_value + recall_value == 0:
        return 0.0
    return 2 * precision_value * recall_value / (precision_value + recall_value)


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _all_pairs(n_fragments: int) -> int:
    count = n_fragments * (n_fragments - 1) // 2
    return max(count, 1)


def _ids_in_pairs(labels: list[PairLabel]) -> set[int]:
    ids: set[int] = set()
    for label in labels:
        ids.add(label.left_id)
        ids.add(label.right_id)
    return ids


def _require_ids(fragments: list[Fragment], needed: set[int]) -> None:
    present = {fragment.id for fragment in fragments}
    missing = sorted(item for item in needed if item not in present)
    if missing:
        raise KeyError(f"label ids not in the store: {missing}")


def _pair_labels(raw: Any, section: str) -> list[PairLabel]:
    if not isinstance(raw, list):
        raise ValueError(f"{section} must be a list")
    labels = []
    for item in raw:
        labels.append(
            PairLabel(
                left_id=_as_int(item, "left_id", section),
                right_id=_as_int(item, "right_id", section),
                positive=_as_bool(item, "positive", section),
            )
        )
    return labels


def _novelty_labels(raw: Any) -> list[NoveltyLabel]:
    if not isinstance(raw, list):
        raise ValueError("novelty must be a list")
    labels = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("novelty entries must be objects")
        similarity = item.get("nearest_similarity")
        if isinstance(similarity, bool) or not isinstance(similarity, (int, float)):
            raise ValueError("nearest_similarity must be a number")
        gold = item.get("gold")
        if not isinstance(gold, str):
            raise ValueError("novelty gold must be a string")
        labels.append(NoveltyLabel(float(similarity), gold))
    return labels


def _critique_labels(raw: Any) -> list[CritiqueLabel]:
    if not isinstance(raw, list):
        raise ValueError("critiques must be a list")
    labels = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("critique entries must be objects")
        scores = {}
        for axis in DIMENSIONS:
            value = item.get(axis)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"critique {axis} must be a number")
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"critique {axis} must be inside [0, 1]")
            scores[axis] = float(value)
        accept = item.get("accept")
        if accept is not None and not isinstance(accept, bool):
            raise ValueError("critique accept must be a boolean")
        labels.append(
            CritiqueLabel(
                candidate_id=_as_int(item, "candidate_id", "critiques"),
                scores=scores,
                accept=accept,
            )
        )
    return labels


def _holdout_cases(raw: Any) -> list[HoldoutCase]:
    if not isinstance(raw, list):
        raise ValueError("holdout must be a list")
    cases = []
    for item in raw:
        cases.append(
            HoldoutCase(
                left_id=_as_int(item, "left_id", "holdout"),
                right_id=_as_int(item, "right_id", "holdout"),
                held_out_id=_as_int(item, "held_out_id", "holdout"),
            )
        )
    return cases


def _as_int(item: Any, field: str, section: str) -> int:
    if not isinstance(item, dict) or field not in item:
        raise ValueError(f"{section} entries need {field}")
    value = item[field]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{section}.{field} must be an integer")
    return value


def _as_bool(item: Any, field: str, section: str) -> bool:
    if not isinstance(item, dict) or field not in item or not isinstance(item[field], bool):
        raise ValueError(f"{section}.{field} must be a boolean")
    return bool(item[field])


def _stored_critiques(store: Store, candidate_ids: list[int]) -> dict[int, dict[str, Any]]:
    found: dict[int, dict[str, Any]] = {}
    for candidate_id in candidate_ids:
        try:
            candidate = store.get_candidate(candidate_id)
        except KeyError:
            continue
        if candidate.critique is None:
            continue
        report = candidate.critique
        found[candidate_id] = {
            "novelty": report.novelty,
            "logical_coherence": report.logical_coherence,
            "falsifiability": report.falsifiability,
            "cross_domain_utility": report.cross_domain_utility,
            "accept": report.accept,
        }
    return found
