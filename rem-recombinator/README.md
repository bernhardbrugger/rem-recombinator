# REM Recombinator

Proof of concept for a "REM sleep"-inspired recombination mechanism:
it searches a personal knowledge store for fragments that are distant
and thematically unrelated, then has an LLM check whether a genuine,
non-obvious insight emerges from them. A second LLM call (the critic)
filters out the noise, which naturally dominates this approach.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in API keys
```

## 1. Import seed data

Export your Claude data under **Settings → Privacy → Export data**. You'll
get an email with several category downloads — you only need the
**"memories"** one (delivered as `memories-000.zip` or as a raw `.json`,
both work).

Try a dry run first (no API keys needed, just shows extraction stats):

```bash
python seed_import.py /path/to/memories-export.json --dry-run
```

Then actually import (needs `ANTHROPIC_API_KEY`/`VOYAGE_API_KEY` in `.env`
for embeddings):

```bash
export $(cat .env | xargs)
python seed_import.py /path/to/memories-export.json
```

Fragments come from three parts of the export: individual `- [stated]`
lines from your curated memory files (highest quality, most atomic),
paragraphs from project memories, and paragraphs from the general
conversations memory.

## 2. Test manually (without MCP)

```python
from recombinator import run_recombination_cycle
from critic import filter_results

raw = run_recombination_cycle()
accepted = filter_results(raw)
for r in accepted:
    print(r["raw_insight"], r["score"])
```

## 3. Use as an MCP server

```bash
python mcp_server.py
```

Register in `claude_desktop_config.json` (see the comment in
`mcp_server.py`).

## Status / open items

- [x] First real export test done — schema confirmed, 314 fragments extracted
      from a real export (290 memory_file / 9 project_memory / 15 conversations_memory)
- [ ] Pairing strategy beyond pure random sampling (e.g. weight by
      access frequency/age — analogous to memories preferentially
      reactivated during sleep)
- [ ] Empirically calibrate the `REM_MAX_SIM` threshold
- [ ] Critic calibration: manually review a sample, tune the score threshold
- [ ] Migrate to Hetzner/Docker once the prototype proves itself
