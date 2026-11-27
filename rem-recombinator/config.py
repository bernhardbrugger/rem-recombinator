"""Compatibility shim. The implementation is :mod:`rem_recombinator.config`."""

from rem_recombinator.config import ConfigurationError, ExperimentConfig, load_config

_LOADED = load_config()
ANTHROPIC_API_KEY = _LOADED.anthropic_api_key
VOYAGE_API_KEY = _LOADED.voyage_api_key
ANTHROPIC_MODEL = _LOADED.anthropic_model
VOYAGE_MODEL = _LOADED.voyage_model
DB_PATH = _LOADED.db_path
MAX_SIMILARITY_FOR_CANDIDATE = _LOADED.max_similarity_for_candidate
CANDIDATES_PER_RUN = _LOADED.candidates_per_run


def validate() -> None:
    """Require the API keys a live Voyage + Anthropic run needs."""
    load_config().validate_keys()


__all__ = [
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_MODEL",
    "CANDIDATES_PER_RUN",
    "ConfigurationError",
    "DB_PATH",
    "ExperimentConfig",
    "MAX_SIMILARITY_FOR_CANDIDATE",
    "VOYAGE_API_KEY",
    "VOYAGE_MODEL",
    "load_config",
    "validate",
]
