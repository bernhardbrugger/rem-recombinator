"""Command line for import, runs, export, stats, lineage, and diagnostics.

User-facing output goes to stdout. Logs go to stderr.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rem_recombinator.config import STRATEGY_CHOICES, ExperimentConfig, load_config
from rem_recombinator.diagnostics import similarity_sample, summarize
from rem_recombinator.embeddings import build_embedder
from rem_recombinator.exporting import render_json, render_markdown
from rem_recombinator.llm import build_language_model
from rem_recombinator.recombinator import execute_run
from rem_recombinator.seed_import import run as import_run
from rem_recombinator.store import Store


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rem-recombinator",
        description="Reproducible semantic recombination of a personal knowledge store.",
    )
    parser.add_argument("--db", type=Path, default=None, help="SQLite path. Default: REM_DB_PATH or ./data/fragments.db")
    parser.add_argument("--verbose", action="store_true", help="Log selection details to stderr")
    sub = parser.add_subparsers(dest="command", required=True)

    import_parser = sub.add_parser("import", help="Import a Claude memories export")
    import_parser.add_argument("export", type=Path)
    import_parser.add_argument("--dry-run", action="store_true")
    import_parser.add_argument("--embedder", choices=("voyage", "hashing"), default="voyage")

    run_parser = sub.add_parser("run", help="Run a recombination experiment")
    run_parser.add_argument("--strategy", choices=STRATEGY_CHOICES, default=None)
    run_parser.add_argument("--n", type=int, default=None)
    run_parser.add_argument("--seed", type=int, default=None)
    run_parser.add_argument("--embedder", choices=("voyage", "hashing"), default="voyage")
    run_parser.add_argument("--llm", choices=("anthropic", "template"), default="anthropic")
    run_parser.add_argument("--no-materialize", action="store_true")

    export_parser = sub.add_parser("export", help="Export runs as Markdown or JSON")
    export_parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    export_parser.add_argument("--output", type=Path, default=None)
    export_parser.add_argument("--accepted-only", action="store_true")

    sub.add_parser("stats", help="Print corpus and critique statistics")

    lineage_parser = sub.add_parser("lineage", help="Show parents and children of a fragment")
    lineage_parser.add_argument("fragment_id", type=int)

    diagnose_parser = sub.add_parser("diagnose", help="Print the pairwise similarity distribution")
    diagnose_parser.add_argument("--samples", type=int, default=2000)
    diagnose_parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args(argv)
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        force=True,
    )
    config = load_config()
    if args.db is not None:
        config = _with_db(config, args.db)
    try:
        if args.command == "import":
            return _import(args, config)
        if args.command == "run":
            return _run(args, config)
        if args.command == "export":
            return _export(args, config)
        if args.command == "stats":
            return _stats(config)
        if args.command == "lineage":
            return _lineage(args, config)
        if args.command == "diagnose":
            return _diagnose(args, config)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    raise AssertionError(f"unhandled command {args.command}")


def _with_db(config: ExperimentConfig, path: Path) -> ExperimentConfig:
    from dataclasses import replace

    return replace(config, db_path=path)


def _import(args: argparse.Namespace, config: ExperimentConfig) -> int:
    if args.dry_run:
        report = import_run(args.export, dry_run=True)
    else:
        embedder = build_embedder(args.embedder, config)
        with Store(config.db_path) as store:
            report = import_run(args.export, store, embedder, dry_run=False)
    print(f"Extracted {report.extracted} fragments")
    for source, count in sorted(report.by_source.items()):
        print(f"  {source}: {count}")
    if args.dry_run:
        print("Dry run: nothing was embedded or stored.")
        for sample in report.samples:
            print(f"  [{sample['source']}] {sample['text']}")
        return 0
    print(f"Added {report.added}. Skipped {report.skipped_existing} already stored.")
    print(f"Database: {config.db_path}")
    return 0


def _run(args: argparse.Namespace, config: ExperimentConfig) -> int:
    embedder = build_embedder(args.embedder, config)
    llm = build_language_model(args.llm, config)
    with Store(config.db_path) as store:
        result = execute_run(
            store=store,
            config=config,
            embedder=embedder,
            llm=llm,
            strategy=args.strategy,
            n=args.n,
            seed=args.seed,
            materialize_accepted=not args.no_materialize,
        )
    print(f"Run {result.run_id}: {len(result.candidates)} candidates, {len(result.accepted)} accepted.")
    for candidate in result.candidates:
        score = None if candidate.critique is None else f"{candidate.critique.composite_score:.3f}"
        print(
            f"- #{candidate.id} {candidate.strategy} {candidate.filter_status} "
            f"parents={candidate.parent_ids} score={score}"
        )
        if candidate.hypothesis:
            print(f"  {candidate.hypothesis}")
    return 0


def _export(args: argparse.Namespace, config: ExperimentConfig) -> int:
    with Store(config.db_path) as store:
        payload = store.export_payload(accepted_only=args.accepted_only)
    rendered = render_json(payload) if args.format == "json" else render_markdown(payload)
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.write_text(rendered, encoding="utf-8")
        print(f"Wrote {args.output}")
    return 0


def _stats(config: ExperimentConfig) -> int:
    with Store(config.db_path) as store:
        stats = store.stats()
    print(f"Fragments: {stats['fragments']}")
    for domain, count in stats["by_domain"].items():
        print(f"  domain {domain}: {count}")
    for source, count in stats["by_source"].items():
        print(f"  source {source}: {count}")
    print(f"Runs: {stats['runs']}")
    print(f"Candidates: {stats['candidates']} (accepted {stats['accepted']})")
    for status, count in stats["by_filter_status"].items():
        print(f"  {status}: {count}")
    if stats["mean_critique"]:
        print("Mean critique (parsed reports):")
        for axis, value in stats["mean_critique"].items():
            print(f"  {axis}: {value:.3f}")
    if stats["legacy_accepted"]:
        print(f"Legacy accepted rows: {stats['legacy_accepted']}")
    return 0


def _lineage(args: argparse.Namespace, config: ExperimentConfig) -> int:
    with Store(config.db_path) as store:
        try:
            payload = store.fragment_lineage(args.fragment_id)
        except KeyError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    fragment = payload["fragment"]
    print(f"Fragment {fragment['id']} [{fragment['domain']}] source={fragment['source']}")
    print(fragment["text"])
    print("Parents: " + (", ".join(str(item["id"]) for item in payload["parents"]) or "none"))
    print("Children: " + (", ".join(str(item["id"]) for item in payload["children"]) or "none"))
    print(
        "Appears in candidates: "
        + (", ".join(str(item) for item in payload["candidate_ids"]) or "none")
    )
    return 0


def _diagnose(args: argparse.Namespace, config: ExperimentConfig) -> int:
    with Store(config.db_path) as store:
        fragments = store.all_fragments()
    print(f"{len(fragments)} fragments in {config.db_path}.")
    if len(fragments) < 2:
        print("Not enough fragments to compare.")
        return 0
    values = similarity_sample(fragments, samples=args.samples, seed=args.seed)
    summary = summarize(values)
    print(f"Similarity distribution over {int(summary['n'])} pairs (seed {args.seed}):")
    for key in ("min", "p10", "p25", "median", "p75", "max"):
        print(f"  {key:7} {summary[key]:.3f}")
    print(
        "A starting REM_MAX_SIM is the p10–p25 band, so bridge pairs come from the "
        "distant tail rather than from an uncalibrated constant."
    )
    return 0
