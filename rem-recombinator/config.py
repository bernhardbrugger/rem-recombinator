"""
Central configuration for the REM Recombinator.
All secrets come from environment variables — never hardcode them.
Loads a local .env file automatically (works the same on Windows,
macOS, and Linux — no manual 'export'/shell-sourcing needed).
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# API Keys
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY", "")

# Models
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
VOYAGE_MODEL = os.environ.get("VOYAGE_EMBED_MODEL", "voyage-3.5")

# Storage
DB_PATH = Path(os.environ.get("REM_DB_PATH", "./data/fragments.db"))

# Recombinator parameters
# Cosine-similarity threshold: below this, two fragments are considered
# "distant" enough to be an interesting REM candidate pair. Tune empirically.
MAX_SIMILARITY_FOR_CANDIDATE = float(os.environ.get("REM_MAX_SIM", "0.35"))

# How many candidate pairs to generate/check per run
CANDIDATES_PER_RUN = int(os.environ.get("REM_CANDIDATES_PER_RUN", "5"))


def validate() -> None:
    missing = []
    if not ANTHROPIC_API_KEY:
        missing.append("ANTHROPIC_API_KEY")
    if not VOYAGE_API_KEY:
        missing.append("VOYAGE_API_KEY")
    if missing:
        raise RuntimeError(
            f"Missing environment variables: {', '.join(missing)}. "
            f"See .env.example."
        )
