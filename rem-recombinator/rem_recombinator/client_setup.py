"""Register the local stdio server with chat clients.

The server reads and writes a SQLite file under the install prefix. This
module never opens that database, never reads an export, and never includes an
API key. The registered process uses the offline hash and the template model
until the user edits the client configuration. Messages are returned to the
caller; this module does not write to stdout.
"""

from __future__ import annotations

import json
from pathlib import Path


def server_spec(python: Path, db_path: Path) -> dict:
    """MCP stdio entry for one private store. No secrets."""
    return {
        "command": str(python),
        "args": ["-m", "rem_recombinator.mcp_server"],
        "env": {
            "REM_DB_PATH": str(db_path),
            "REM_EMBEDDER": "hashing",
            "REM_LLM": "template",
        },
    }


def register_clients(
    prefix: Path,
    home: Path,
    *,
    python: Path | None = None,
    clients: bool = True,
) -> tuple[list[Path], list[str]]:
    """Merge the server into client configs.

    Returns written paths and warnings. An existing file that is not valid JSON
    is left untouched and named in the warnings.
    """
    prefix = prefix.resolve()
    home = home.resolve()
    # absolute(), not resolve(): a venv interpreter is a symlink to the system
    # Python, and launching the system path drops the virtualenv.
    python_path = (python or prefix / "venv" / "bin" / "python").absolute()
    db_path = prefix / "state" / "fragments.db"
    spec = server_spec(python_path, db_path)
    written: list[Path] = []
    warnings: list[str] = []

    snippet = prefix / "mcp-snippet.json"
    _write_json(snippet, {"mcpServers": {"rem-recombinator": spec}})
    written.append(snippet)
    if not clients:
        return written, warnings

    cursor = home / ".cursor" / "mcp.json"
    _take(cursor, spec, written, warnings)
    claude = _claude_config(home)
    if claude is not None:
        _take(claude, spec, written, warnings)
    return written, warnings


def registration_message(prefix: Path, written: list[Path], warnings: list[str]) -> str:
    """Text the installer shows. No secret values."""
    db_path = prefix / "state" / "fragments.db"
    lines = [
        "Private store (created on first use, never uploaded):",
        f"  {db_path}",
        "Registered:",
        *[f"  {path}" for path in written],
        *warnings,
        "Restart the chat client so it reloads MCP.",
        "The server does not read your chats. Import a local file when you want a corpus:",
        "  rem-recombinator import ./fragments.json --embedder hashing",
        '  {"fragments": [{"text": "...", "domain": "physics"}]}',
        "A directory of .md or .txt files works too. The folder name becomes the domain.",
        "The public fictional run uses a temporary database and does not touch the store above:",
        "  rem-recombinator demo",
        "This install calls neither Anthropic nor Voyage. Fragment text stays on this machine",
        "until you change REM_EMBEDDER or REM_LLM in the client config and add API keys.",
    ]
    return "\n".join(lines)


def _claude_config(home: Path) -> Path | None:
    candidates = [
        home / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json",
        home / ".config" / "Claude" / "claude_desktop_config.json",
    ]
    for path in candidates:
        if path.exists() or path.parent.exists():
            return path
    return None


def _take(path: Path, spec: dict, written: list[Path], warnings: list[str]) -> None:
    status = _merge_mcp(path, spec)
    if status == "written":
        written.append(path)
    elif status:
        warnings.append(status)


def _merge_mcp(path: Path, spec: dict) -> str:
    """Insert rem-recombinator. Return ``written`` or a warning. Never clobber invalid JSON."""
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8") or "{}")
        except json.JSONDecodeError:
            return f"left unchanged (not valid JSON): {path}"
        if not isinstance(data, dict):
            return f"left unchanged (JSON value is not an object): {path}"
    else:
        data = {}
    servers = data.get("mcpServers")
    if servers is None:
        servers = {}
        data["mcpServers"] = servers
    if not isinstance(servers, dict):
        return f"left unchanged (mcpServers is not an object): {path}"
    servers["rem-recombinator"] = spec
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, data)
    return "written"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
