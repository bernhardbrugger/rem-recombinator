# REM Recombinator

A rejection-heavy procedure for a private knowledge store. Pairs of fragments are chosen by an explicit geometric rule, a model may propose one hypothesis, and the code discards it unless it clears a novelty window and a four-axis critic. The name is a sketch of REM sleep, in which weakly linked traces are reactivated together and most of those collisions are noise. It is not a claim about brains.

The public note draws the method on seven fictional sentences, including the animated geometry:

**[bernhardbrugger.github.io/rem-recombinator](https://bernhardbrugger.github.io/rem-recombinator/)**

Personal chats are not in this repository, not in the nightly artifact, and not on that page.

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/bernhardbrugger/rem-recombinator/main/install.sh | bash
```

The command clones the program under `~/.local/share/rem-recombinator`, builds a virtualenv, and registers a local stdio server with Cursor or Claude Desktop when that client's config directory exists. Nothing is uploaded. No API key is required. The registered embedder is the offline hash, and the registered model is the template, which accepts nothing. That is the gate, not a result about your notes.

```bash
rem-recombinator import ./notes/ --embedder hashing
rem-recombinator demo
```

`demo` reruns the fictional sample in a temporary database. A directory of notes uses the folder name as the domain. Fragment text leaves the machine only if you later point `REM_EMBEDDER` or `REM_LLM` at an API and add keys locally.

## What the code decides

Cosine similarity, with a zero vector defined as similarity 0:

```text
cos(u, v) = (u · v) / (||u|| ||v||)
```

| Rule | A pair is kept when | Score |
| --- | --- | --- |
| Semantic bridge | Different clusters, cosine in [0.05, 0.35] | Novelty of the normalised midpoint |
| Analogical transfer | Different domains, cosine in [0.15, 0.55] | The cosine |
| Dialectic synthesis | A listed opposition, cosine in [0.20, 0.75] | Tension times closeness to 0.45 |

A hypothesis is dropped when its nearest neighbour is above 0.88 or below 0.05. The critic scores novelty, logical coherence, falsifiability, and cross-domain utility in [0, 1]. The report is accepted only when

```text
C = 0.30 N + 0.30 L + 0.20 F + 0.20 U
```

is at least 0.60 and every axis is at least 0.30. The model does not set the accept bit.

## Where to read on

The manual, with the data model, the worked numerical example, the MCP tools, and the limits of the method, is [`rem-recombinator/README.md`](rem-recombinator/README.md). The package is [`rem-recombinator/`](rem-recombinator/).

A GitHub Action runs the same fictional cycle every night at 02:15 UTC (03:15 Europe/Berlin in winter, 04:15 in summer) and uploads `nightly-out/` as an artifact. It does not grow a personal memory. See [Nightly run](rem-recombinator/README.md#nightly-run).

## Develop

```bash
cd rem-recombinator
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
pytest
python -m rem_recombinator import examples/sample_memory_export.json --embedder hashing
python -m rem_recombinator evaluate examples/evaluation_labels.example.json
```

`evaluate` reports precision and recall for the similarity thresholds. It does not change the defaults. `.env`, `*.db`, and `memories/*.json` are gitignored.
