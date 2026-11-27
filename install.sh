#!/usr/bin/env bash
# Install REM Recombinator into the home directory and register a local MCP server.
# The script does not read chats, does not import a personal export, and does not
# upload anything. The registered model is the offline template.
set -euo pipefail

PREFIX="${REM_PREFIX:-$HOME/.local/share/rem-recombinator}"
SRC="$PREFIX/src"
VENV="$PREFIX/venv"
STATE="$PREFIX/state"
BIN_DIR="${REM_BIN_DIR:-$HOME/.local/bin}"
REPO="${REM_REPO_URL:-https://github.com/bernhardbrugger/rem-recombinator.git}"
REF="${REM_REF:-main}"

mkdir -p "$STATE" "$BIN_DIR" "$(dirname "$SRC")"

if [[ -n "${REM_SOURCE_DIR:-}" ]]; then
  rm -rf "$SRC"
  mkdir -p "$SRC"
  cp -a "$REM_SOURCE_DIR"/. "$SRC"/
else
  if [[ -d "$SRC/.git" ]]; then
    git -C "$SRC" fetch --depth 1 origin "$REF"
    git -C "$SRC" checkout --force FETCH_HEAD
  else
    rm -rf "$SRC"
    git clone --depth 1 --branch "$REF" "$REPO" "$SRC"
  fi
fi

python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install -e "$SRC/rem-recombinator"
ln -sfn "$VENV/bin/rem-recombinator" "$BIN_DIR/rem-recombinator"

REGISTER=(register --prefix "$PREFIX" --python "$VENV/bin/python")
if [[ "${REM_NO_REGISTER:-}" == "1" ]]; then
  REGISTER+=(--no-register)
fi
"$VENV/bin/python" -m rem_recombinator "${REGISTER[@]}"

if ! command -v rem-recombinator >/dev/null 2>&1; then
  echo "Add this directory to PATH if the command is not found:"
  echo "  export PATH=\"$BIN_DIR:\$PATH\""
fi
