"""Experiment configuration.

Every value that can change a result is read here and later copied into the
``experiment_runs.hyperparameters`` column. API keys are never part of that
snapshot.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

STRATEGIES: tuple[str, ...] = (
    "semantic_bridge",
    "analogical_transfer",
    "dialectic_synthesis",
)
STRATEGY_CHOICES: tuple[str, ...] = (*STRATEGIES, "all")

# voyage-3.5 default width. The store records the actual vector length.
DEFAULT_EMBEDDING_DIM = 1024
ALLOWED_VOYAGE_DIMS = frozenset({256, 512, 1024, 2048})

_SECRET_FIELDS = frozenset({"anthropic_api_key", "voyage_api_key"})


class ConfigurationError(ValueError):
    """Raised when environment or threshold settings cannot form a valid run."""


def _load_env_files() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(override=False)
    project_env = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(project_env, override=False)


def _raw(name: str, default: str, environ: dict[str, str]) -> str:
    value = environ.get(name)
    if value is None or value == "":
        return default
    return value


def _as_float(name: str, default: str, environ: dict[str, str]) -> float:
    raw = _raw(name, default, environ)
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a float, got {raw!r}") from exc


def _as_int(name: str, default: str, environ: dict[str, str]) -> int:
    raw = _raw(name, default, environ)
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer, got {raw!r}") from exc


@dataclass(frozen=True)
class ExperimentConfig:
    """Immutable snapshot of one run's controls."""

    anthropic_api_key: str = ""
    voyage_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-6"
    voyage_model: str = "voyage-3.5"
    embedding_dimensions: int = DEFAULT_EMBEDDING_DIM
    temperature: float = 0.4
    critic_temperature: float = 0.0
    seed: int = 0
    db_path: Path = Path("./data/fragments.db")
    max_similarity_for_candidate: float = 0.35
    min_similarity_for_candidate: float = 0.05
    analogical_min_similarity: float = 0.15
    analogical_max_similarity: float = 0.55
    dialectic_min_similarity: float = 0.20
    dialectic_max_similarity: float = 0.75
    dialectic_target_similarity: float = 0.45
    cluster_link_threshold: float = 0.55
    novelty_ceiling: float = 0.88
    incoherence_floor: float = 0.05
    candidates_per_run: int = 5
    pair_enumeration_cap: int = 8000
    strategy: str = "semantic_bridge"
    critic_accept_threshold: float = 0.60
    critic_dimension_floor: float = 0.30
    weight_novelty: float = 0.30
    weight_coherence: float = 0.30
    weight_falsifiability: float = 0.20
    weight_utility: float = 0.20
    max_tokens: int = 800

    def validate_keys(self, *, need_anthropic: bool = True, need_voyage: bool = True) -> None:
        missing: list[str] = []
        if need_anthropic and not self.anthropic_api_key:
            missing.append("ANTHROPIC_API_KEY")
        if need_voyage and not self.voyage_api_key:
            missing.append("VOYAGE_API_KEY")
        if missing:
            raise ConfigurationError(
                "Missing environment variables: "
                + ", ".join(missing)
                + ". Copy .env.example to .env and fill it in. .env is gitignored."
            )

    def validate_ranges(self) -> None:
        if self.strategy not in STRATEGY_CHOICES:
            raise ConfigurationError(
                f"strategy must be one of {', '.join(STRATEGY_CHOICES)}, got {self.strategy!r}"
            )
        if self.seed < 0:
            raise ConfigurationError("seed must be >= 0")
        if self.candidates_per_run < 1:
            raise ConfigurationError("candidates_per_run must be >= 1")
        if self.pair_enumeration_cap < 1:
            raise ConfigurationError("pair_enumeration_cap must be >= 1")
        if self.embedding_dimensions not in ALLOWED_VOYAGE_DIMS:
            raise ConfigurationError(
                "embedding_dimensions must be one of "
                + ", ".join(str(d) for d in sorted(ALLOWED_VOYAGE_DIMS))
            )
        for name in ("temperature", "critic_temperature"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ConfigurationError(f"{name} must be between 0 and 1")
        bands = (
            ("min_similarity_for_candidate", "max_similarity_for_candidate"),
            ("analogical_min_similarity", "analogical_max_similarity"),
            ("dialectic_min_similarity", "dialectic_max_similarity"),
        )
        for low_name, high_name in bands:
            low, high = getattr(self, low_name), getattr(self, high_name)
            if not 0.0 <= low < high <= 1.0:
                raise ConfigurationError(f"require 0 <= {low_name} < {high_name} <= 1")
        if not self.dialectic_min_similarity <= self.dialectic_target_similarity <= self.dialectic_max_similarity:
            raise ConfigurationError("dialectic_target_similarity must lie inside the dialectic band")
        if not 0.0 < self.incoherence_floor < self.novelty_ceiling < 1.0:
            raise ConfigurationError("require 0 < incoherence_floor < novelty_ceiling < 1")
        if not 0.0 <= self.cluster_link_threshold <= 1.0:
            raise ConfigurationError("cluster_link_threshold must be between 0 and 1")
        if not 0.0 <= self.critic_accept_threshold <= 1.0:
            raise ConfigurationError("critic_accept_threshold must be between 0 and 1")
        if not 0.0 <= self.critic_dimension_floor <= 1.0:
            raise ConfigurationError("critic_dimension_floor must be between 0 and 1")
        weights = self.weights()
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ConfigurationError(f"critic weights must sum to 1, got {total:.6f}")
        if any(value < 0 for value in weights.values()):
            raise ConfigurationError("critic weights must be non-negative")

    def weights(self) -> dict[str, float]:
        return {
            "novelty": self.weight_novelty,
            "logical_coherence": self.weight_coherence,
            "falsifiability": self.weight_falsifiability,
            "cross_domain_utility": self.weight_utility,
        }

    def public_dict(self) -> dict[str, object]:
        """Hyperparameters safe to persist. Secrets are omitted."""
        raw = asdict(self)
        for key in _SECRET_FIELDS:
            raw.pop(key, None)
        raw["db_path"] = str(self.db_path)
        raw["weights"] = self.weights()
        return raw


def load_config(environ: dict[str, str] | None = None) -> ExperimentConfig:
    """Build a config from the process environment.

    ``environ`` exists so tests can pass a dict without mutating ``os.environ``.
    When it is omitted, ``.env`` is loaded first and the real environment is read.
    """
    if environ is None:
        _load_env_files()
        source = dict(os.environ)
    else:
        source = environ
    config = ExperimentConfig(
        anthropic_api_key=_raw("ANTHROPIC_API_KEY", "", source),
        voyage_api_key=_raw("VOYAGE_API_KEY", "", source),
        anthropic_model=_raw("ANTHROPIC_MODEL", "claude-sonnet-4-6", source),
        voyage_model=_raw("VOYAGE_EMBED_MODEL", "voyage-3.5", source),
        embedding_dimensions=_as_int("REM_EMBED_DIM", "1024", source),
        temperature=_as_float("REM_TEMPERATURE", "0.4", source),
        critic_temperature=_as_float("REM_CRITIC_TEMPERATURE", "0.0", source),
        seed=_as_int("REM_SEED", "0", source),
        db_path=Path(_raw("REM_DB_PATH", "./data/fragments.db", source)),
        max_similarity_for_candidate=_as_float("REM_MAX_SIM", "0.35", source),
        min_similarity_for_candidate=_as_float("REM_MIN_SIM", "0.05", source),
        analogical_min_similarity=_as_float("REM_ANALOGICAL_MIN", "0.15", source),
        analogical_max_similarity=_as_float("REM_ANALOGICAL_MAX", "0.55", source),
        dialectic_min_similarity=_as_float("REM_DIALECTIC_MIN", "0.20", source),
        dialectic_max_similarity=_as_float("REM_DIALECTIC_MAX", "0.75", source),
        dialectic_target_similarity=_as_float("REM_DIALECTIC_TARGET", "0.45", source),
        cluster_link_threshold=_as_float("REM_CLUSTER_LINK", "0.55", source),
        novelty_ceiling=_as_float("REM_NOVELTY_CEILING", "0.88", source),
        incoherence_floor=_as_float("REM_INCOHERENCE_FLOOR", "0.05", source),
        candidates_per_run=_as_int("REM_CANDIDATES_PER_RUN", "5", source),
        pair_enumeration_cap=_as_int("REM_PAIR_CAP", "8000", source),
        strategy=_raw("REM_STRATEGY", "semantic_bridge", source),
        critic_accept_threshold=_as_float("REM_CRITIC_THRESHOLD", "0.60", source),
        critic_dimension_floor=_as_float("REM_CRITIC_FLOOR", "0.30", source),
        weight_novelty=_as_float("REM_WEIGHT_NOVELTY", "0.30", source),
        weight_coherence=_as_float("REM_WEIGHT_COHERENCE", "0.30", source),
        weight_falsifiability=_as_float("REM_WEIGHT_FALSIFIABILITY", "0.20", source),
        weight_utility=_as_float("REM_WEIGHT_UTILITY", "0.20", source),
        max_tokens=_as_int("REM_MAX_TOKENS", "800", source),
    )
    config.validate_ranges()
    return config


def embedder_name(environ: dict[str, str] | None = None) -> str:
    """Which embedder the local server should construct.

    An explicit ``REM_EMBEDDER`` wins. Otherwise a Voyage key selects the
    remote model, and a missing key stays on the offline hash. The hash is
    the default for a fresh install so a chat client can call the tools
    without sending text to an embedding API.
    """
    source = os.environ if environ is None else environ
    explicit = source.get("REM_EMBEDDER", "").strip()
    if explicit:
        return explicit
    if source.get("VOYAGE_API_KEY", "").strip():
        return "voyage"
    return "hashing"


def llm_name(environ: dict[str, str] | None = None) -> str:
    """Which language model the local server should construct.

    An explicit ``REM_LLM`` wins. Otherwise an Anthropic key selects the
    remote model, and a missing key stays on the offline template. The
    template does not call a network.
    """
    source = os.environ if environ is None else environ
    explicit = source.get("REM_LLM", "").strip()
    if explicit:
        return explicit
    if source.get("ANTHROPIC_API_KEY", "").strip():
        return "anthropic"
    return "template"
