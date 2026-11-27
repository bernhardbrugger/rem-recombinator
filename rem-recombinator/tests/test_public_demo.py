"""Public corpus, neutral import, and local client registration. No network."""

import json
from pathlib import Path

from rem_recombinator.cli import main
from rem_recombinator.client_setup import register_clients, server_spec
from rem_recombinator.config import embedder_name, llm_name
from rem_recombinator.public_corpus import build_public_geometry
from rem_recombinator.seed_import import load_fragment_records

ROOT = Path(__file__).resolve().parents[2]
GEOMETRY = ROOT / "docs" / "geometry.json"


def test_backend_defaults_stay_offline_without_keys():
    assert embedder_name({}) == "hashing"
    assert llm_name({}) == "template"
    assert embedder_name({"REM_EMBEDDER": "voyage", "VOYAGE_API_KEY": ""}) == "voyage"
    assert llm_name({"ANTHROPIC_API_KEY": "present"}) == "anthropic"
    assert llm_name({"ANTHROPIC_API_KEY": "present", "REM_LLM": "template"}) == "template"


def test_generic_json_and_directory(tmp_path):
    payload = {
        "fragments": [
            {"text": "too short", "domain": "physics"},
            {"text": "Heat always increases entropy in an isolated system.", "domain": "physics"},
            {"text": "Selection can shrink diversity inside one niche.", "domain": " biology "},
        ]
    }
    path = tmp_path / "notes.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    records = load_fragment_records(path)
    assert [record["domain"] for record in records] == ["physics", "biology"]
    assert records[0]["source"] == "fragment"

    folder = tmp_path / "corpus"
    (folder / "physics").mkdir(parents=True)
    (folder / "physics" / "note.md").write_text(
        "Closed systems always increase entropy when heat flows in and no work leaves.\n\n"
        "Open systems can export entropy and so decrease it locally.\n",
        encoding="utf-8",
    )
    (folder / ".hidden").mkdir()
    (folder / ".hidden" / "secret.md").write_text("This hidden note must not be imported at all.\n\n" * 2, encoding="utf-8")
    from_dir = load_fragment_records(folder)
    assert len(from_dir) == 2
    assert {record["domain"] for record in from_dir} == {"physics"}
    assert all("hidden" not in record["text"] for record in from_dir)


def test_published_geometry_matches_the_generator():
    fresh = build_public_geometry()
    published = json.loads(GEOMETRY.read_text(encoding="utf-8"))
    assert published == fresh
    assert len(fresh["fragments"]) == 7
    assert fresh["embedder"] == "hashing-32"
    assert fresh["thresholds"]["accept"] == 0.6
    assert fresh["thresholds"]["novelty_ceiling"] == 0.88
    assert {item["domain"] for item in fresh["fragments"]} == {"physics", "biology", "project", "conversation"}


def test_demo_uses_a_temporary_database(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["demo"]) == 0
    out = capsys.readouterr().out
    assert "Fictional public corpus" in out
    assert "0 accepted" in out
    assert not (tmp_path / "data").exists()
    assert list(tmp_path.iterdir()) == []


def test_register_merges_cursor_and_skips_broken_json(tmp_path):
    home = tmp_path / "home"
    prefix = tmp_path / "prefix"
    (prefix / "state").mkdir(parents=True)
    cursor = home / ".cursor" / "mcp.json"
    cursor.parent.mkdir(parents=True)
    cursor.write_text(json.dumps({"mcpServers": {"other": {"command": "other"}}}), encoding="utf-8")
    claude_dir = home / ".config" / "Claude"
    claude_dir.mkdir(parents=True)
    broken = claude_dir / "claude_desktop_config.json"
    broken.write_text("{", encoding="utf-8")

    written, warnings = register_clients(prefix, home, python=prefix / "venv" / "bin" / "python")
    assert any("not valid JSON" in warning for warning in warnings)
    saved = json.loads(cursor.read_text(encoding="utf-8"))
    assert set(saved["mcpServers"]) == {"other", "rem-recombinator"}
    spec = saved["mcpServers"]["rem-recombinator"]
    assert spec["env"]["REM_EMBEDDER"] == "hashing"
    assert spec["env"]["REM_LLM"] == "template"
    assert "KEY" not in json.dumps(spec)
    assert broken.read_text(encoding="utf-8") == "{"
    assert prefix / "mcp-snippet.json" in written
    assert server_spec(prefix / "python", prefix / "state" / "fragments.db")["args"] == [
        "-m",
        "rem_recombinator.mcp_server",
    ]


def test_register_keeps_the_venv_interpreter_symlink(tmp_path):
    home = tmp_path / "home"
    prefix = tmp_path / "prefix"
    bin_dir = prefix / "venv" / "bin"
    bin_dir.mkdir(parents=True)
    target = tmp_path / "system-python"
    target.write_text("", encoding="utf-8")
    link = bin_dir / "python"
    link.symlink_to(target)
    register_clients(prefix, home, python=link)
    saved = json.loads((home / ".cursor" / "mcp.json").read_text(encoding="utf-8"))
    command = saved["mcpServers"]["rem-recombinator"]["command"]
    assert command == str(link.absolute())
    assert not command.endswith("system-python")
