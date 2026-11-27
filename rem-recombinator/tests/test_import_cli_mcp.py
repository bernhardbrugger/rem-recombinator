"""Seed import, CLI, and MCP handlers. No network and no stdout from the server."""

import json
import zipfile
from pathlib import Path

from rem_recombinator.cli import main
from rem_recombinator.mcp_server import dispatch, tool_specs, validate_arguments
from rem_recombinator.seed_import import domain_from_memory_path, extract_fragments, load_export
from tests.fakes import MapEmbedder, unit

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "sample_memory_export.json"


def test_extracts_stated_lines_paragraphs_and_domains():
    fragments = extract_fragments(json.loads(EXAMPLE.read_text(encoding="utf-8")))
    by_source = {}
    for fragment in fragments:
        by_source[fragment["source"]] = by_source.get(fragment["source"], 0) + 1
    assert by_source["memory_file"] == 3
    assert by_source["project_memory"] == 2
    assert by_source["conversations_memory"] == 2
    domains = {fragment["domain"] for fragment in fragments if fragment["source"] == "memory_file"}
    assert domains == {"physics", "biology"}
    assert domain_from_memory_path("/physics/thermo.md") == "physics"
    assert domain_from_memory_path("note.md") == "memory"


def test_zip_and_resume(store, tmp_path):
    archive_path = tmp_path / "memories.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.write(EXAMPLE, arcname="memories.json")
    data = load_export(archive_path)
    fragments = extract_fragments(data)
    mapping = {fragment["text"]: unit(1, (index + 1) / 10, 0) for index, fragment in enumerate(fragments)}
    embedder = MapEmbedder(mapping)
    from rem_recombinator.seed_import import import_extracted

    first = import_extracted(fragments, store, embedder)
    second = import_extracted(fragments, store, embedder)
    assert first.added == len(fragments)
    assert second.added == 0
    assert second.skipped_existing == len(fragments)
    assert store.count_fragments() == len(fragments)


def test_cli_import_run_stats_and_export(tmp_path, capsys):
    database = tmp_path / "cli.db"
    code = main(["--db", str(database), "import", str(EXAMPLE), "--dry-run"])
    assert code == 0
    assert "Dry run" in capsys.readouterr().out

    code = main(
        ["--db", str(database), "import", str(EXAMPLE), "--embedder", "hashing"]
    )
    assert code == 0
    assert "Added" in capsys.readouterr().out

    code = main(
        [
            "--db",
            str(database),
            "run",
            "--embedder",
            "hashing",
            "--llm",
            "template",
            "--strategy",
            "semantic_bridge",
            "--n",
            "1",
            "--seed",
            "0",
        ]
    )
    assert code == 0
    assert "Run 1" in capsys.readouterr().out

    code = main(["--db", str(database), "stats"])
    assert code == 0
    assert "Fragments:" in capsys.readouterr().out

    destination = tmp_path / "out.md"
    code = main(["--db", str(database), "export", "--format", "markdown", "--output", str(destination)])
    assert code == 0
    text = destination.read_text(encoding="utf-8")
    assert "REM recombination export" in text
    assert "sk-ant" not in text


def test_mcp_validation_and_stdout(store, config, capsys):
    names = [spec["name"] for spec in tool_specs()]
    assert names == [
        "run_rem_cycle",
        "list_accepted_insights",
        "store_stats",
        "fragment_lineage",
    ]
    try:
        validate_arguments("run_rem_cycle", {"strategy": "nope"})
    except ValueError as exc:
        assert "strategy" in str(exc)
    else:
        raise AssertionError("bad strategy must fail validation")
    try:
        validate_arguments("run_rem_cycle", {"candidates": True})
    except ValueError as exc:
        assert "integer" in str(exc)
    else:
        raise AssertionError("booleans are not integers here")
    try:
        validate_arguments("fragment_lineage", {})
    except ValueError as exc:
        assert "fragment_id" in str(exc)
    else:
        raise AssertionError("fragment_id is required")

    text = dispatch("store_stats", {}, store=store, config=config)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Fragments: 0" in text
    missing = dispatch("fragment_lineage", {"fragment_id": 9}, store=store, config=config)
    assert "No fragment 9" in missing
    assert capsys.readouterr().out == ""

    left = "Closed systems spread heat."
    right = "Crystal lattices tile space."
    store.add_fragment(left, unit(1, 0, 0), source="test", domain="physics")
    store.add_fragment(right, unit(0.25, (1 - 0.25**2) ** 0.5, 0), source="test", domain="physics")
    hypothesis = "A bookkeeping link between heat and tiling."
    from rem_recombinator.llm import ScriptedLanguageModel
    from rem_recombinator.critic import DIMENSIONS

    model = ScriptedLanguageModel(
        hypotheses=[{"hypothesis": hypothesis, "rationale": "Shared accounting.", "no_genuine_link": False}],
        critiques=[
            {
                "novelty": 0.9,
                "logical_coherence": 0.9,
                "falsifiability": 0.9,
                "cross_domain_utility": 0.9,
                "rationales": {name: "Specific." for name in DIMENSIONS},
                "improvement": "State the observable.",
            }
        ],
    )
    embedder = MapEmbedder(
        {
            left: unit(1, 0, 0),
            right: unit(0.25, (1 - 0.25**2) ** 0.5, 0),
            hypothesis: unit(0.2, 0.2, 0.96),
        }
    )
    report = dispatch(
        "run_rem_cycle",
        {"strategy": "semantic_bridge", "candidates": 1, "seed": 0, "materialize": True},
        store=store,
        config=config,
        embedder=embedder,
        llm=model,
    )
    assert capsys.readouterr().out == ""
    assert "accepted" in report
    listed = dispatch("list_accepted_insights", {"limit": 5}, store=store, config=config)
    assert "bookkeeping" in listed
    assert capsys.readouterr().out == ""


def test_mcp_module_source_has_no_print():
    source = Path(__file__).resolve().parents[1].joinpath("rem_recombinator", "mcp_server.py").read_text(
        encoding="utf-8"
    )
    assert "print(" not in source
