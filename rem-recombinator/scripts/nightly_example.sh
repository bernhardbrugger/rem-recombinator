#!/usr/bin/env bash
# One recombination cycle on the fictional sample export.
# The database and the report stay under nightly-out/ and are gitignored.
# Hashing + the template model need no API keys. A live night sets
# REM_EMBEDDER=voyage and REM_LLM=anthropic, plus the two keys.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT="${NIGHTLY_OUT:-$ROOT/nightly-out}"
mkdir -p "$OUT"
export REM_DB_PATH="${REM_DB_PATH:-$OUT/fragments.db}"
export REM_SEED="${REM_SEED:-0}"

EMBEDDER="${REM_EMBEDDER:-hashing}"
LLM="${REM_LLM:-template}"
STRATEGY="${REM_STRATEGY:-all}"
N="${NIGHTLY_CANDIDATES:-3}"
PY="${PYTHON:-python3}"

echo "cycle embedder=$EMBEDDER llm=$LLM strategy=$STRATEGY n=$N seed=$REM_SEED"
echo "database $REM_DB_PATH"

"$PY" -m rem_recombinator import examples/sample_memory_export.json --embedder "$EMBEDDER"
"$PY" -m rem_recombinator --verbose run \
  --embedder "$EMBEDDER" \
  --llm "$LLM" \
  --strategy "$STRATEGY" \
  --n "$N" \
  --seed "$REM_SEED"
"$PY" -m rem_recombinator stats | tee "$OUT/stats.txt"
"$PY" -m rem_recombinator export --format markdown --output "$OUT/nightly.md"
"$PY" -m rem_recombinator export --format json --output "$OUT/nightly.json"

echo "wrote $OUT/stats.txt"
echo "wrote $OUT/nightly.md"
echo "wrote $OUT/nightly.json"
