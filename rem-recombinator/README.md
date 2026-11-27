# REM Recombinator

REM Recombinator is a small research framework for **semantic recombination** of a personal knowledge store. The name points at a computational sketch of REM sleep, not at a claim that the program sleeps or consolidates memory the way a brain does: during REM, weakly linked traces are reactivated together, and most of those collisions are noise. The system does the analogous thing on purpose. It selects fragment pairs by an explicit rule, asks a model for one hypothesis, throws the hypothesis away when its embedding is too close to the corpus or has no anchor in it, and scores what remains on four axes. Every choice is written to SQLite with the hyperparameters that produced it.

Run the commands below from this directory (`rem-recombinator/`).

## Architecture

```
memory export (.json / .zip)
        |
        v
 seed_import.extract_fragments        domain tag + source + origin
        |
        v
 Embedder.embed_batch                 voyage-3.5 (1024-d) or hashing (offline)
        |
        v
 Store (SQLite)                       fragments, runs, candidates, critiques, lineage
        |
        +-- strategies.select_pairs   semantic bridge | analogical transfer | dialectic
        |
        v
 LanguageModel.recombine              forced tool call, separate from the critic
        |
        v
 novelty.assess_novelty               drop if max cosine > 0.88 or < 0.05
        |
        v
 LanguageModel.critique               four scores; the code decides accept
        |
        v
 lineage edge                         accepted hypothesis becomes a fragment
        |
        +-- cli / MCP (stdio)         stdout is the transport; logs go to stderr
```

The language model and the embedder are injected. Tests use a map from strings to vectors and a scripted model, so the suite never calls a network.

## Data model

| Object | What it records |
| --- | --- |
| `Fragment` | text, source, domain tag, timestamp, float32 embedding, width, JSON metadata |
| `RecombinationCandidate` | `parent_ids`, strategy, hypothesis, rationale, pair cosine, novelty, filter status, parent snapshot |
| `CritiqueReport` | novelty, logical coherence, falsifiability, cross-domain utility, a rationale per axis, one improvement, composite, accept |
| `experiment_runs` | seed, strategy, and the public hyperparameter snapshot |

Provenance is stored three ways, so a generated concept can be walked back to its parents even if a later edit changes a row:

- `candidate_parents` links the candidate to fragment ids, in order.
- `parent_snapshot` copies id, domain, source, and text at generation time.
- `lineage` links a materialized child fragment to those parents (`relation = recombination_parent`).

Accept is **not** a field the model is trusted to set. The pipeline computes it. Invalid critic JSON is stored as a rejected report with `parse_error`, so a broken call is still part of the experiment.

Opening a database written by the first prototype adds `domain` and `embedding_dim`. Existing rows take their domain from `source`. The old `recombinations` table stays readable and shows up in exports as `legacy_accepted`.

## Selection

Cosine similarity, with a zero vector defined as similarity 0:

```
cos(u, v) = (u · v) / (||u|| ||v||)
```

Pairs are enumerated in id order when their number is at most `REM_PAIR_CAP` (default 8000). Larger corpora take a seeded sample. Ties break by fragment id, so a small corpus does not depend on the seed.

### Clusters

Semantic bridge needs a notion of region. The implementation is greedy leader clustering, not a fitted mixture:

1. Walk fragments by ascending id.
2. A fragment becomes a leader when its cosine to every existing leader is strictly below `REM_CLUSTER_LINK` (default 0.55).
3. Otherwise it joins the nearest leader.

The partition depends on id order. It is reproducible. It is not a globally optimal clustering. That limit is deliberate: there is no extra dependency, and the rule is short enough to test.

### Semantic bridge

Eligible pairs sit in **different** clusters and in the band

```
REM_MIN_SIM ≤ cos(u, v) ≤ REM_MAX_SIM
```

(defaults 0.05 and 0.35). The score is the novelty of the equal-weight midpoint. Each parent is L2-normalized before the sum, so a longer raw vector does not dominate:

```
m = normalize(normalize(u) + normalize(v))
S = 1 - max_{k not in {i, j}} cos(m, v_k)
```

A pair whose midpoint lands on top of some third fragment ranks last. Orthogonal pairs never enter the band, because a midpoint between unrelated directions is not a bridge.

### Analogical transfer

Eligible pairs have **different** domain tags, neither of them `unspecified`, and a cosine in `[REM_ANALOGICAL_MIN, REM_ANALOGICAL_MAX]` (0.15, 0.55). Inside that band the score **is** the cosine: a shared direction across domains is the analogy. Fragment A is the lexicographically smaller domain (the source); Fragment B is the target.

### Dialectic synthesis

Eligible pairs have cosine in `[REM_DIALECTIC_MIN, REM_DIALECTIC_MAX]` (0.20, 0.75) and at least one lexical opposition from a fixed cue list (`always`/`never`, `increase`/`decrease`, `confirm`/`disconfirm`, …). Matching is on word boundaries, so `instrument` does not count as `true`. The score is

```
tension * (1 - |cos(u, v) - REM_DIALECTIC_TARGET|)
```

with target 0.45. This is a cue heuristic, not a proof that the sentences contradict.

### Worked geometry

Three unit vectors, default thresholds:

| id | domain | text (short) | vector |
| --- | --- | --- | --- |
| A | physics | Heat always increases entropy in an isolated system. | `(1, 0, 0)` |
| B | physics | Heat never increases entropy once the system can export it. | `normalize(0.45, √(1−0.45²), 0)` |
| C | biology | Selection pressure can shrink diversity inside a local niche. | `normalize(0.2, 0.2, 0.96)` |

| pair | cosine | same cluster? | bridge | analogy | dialectic |
| --- | --- | --- | --- | --- | --- |
| A, B | 0.4500 | no (0.45 < 0.55) | no, above 0.35 | no, same domain | yes, tension 1, score 1.000 |
| A, C | 0.1998 | no | yes, score 0.536 | yes, biology → physics, score 0.200 | no, below 0.20 and no cue |
| B, C | 0.2684 | no | yes, score 0.592 | yes, biology → physics, score 0.268 | no cue |

Bridge therefore returns B–C, then A–C. Dialectic returns only A–B. Analogy returns C–B, then C–A, with biology as the source domain. These numbers come from `tests/test_strategies.py` and the same functions the CLI calls.

## Novelty filter

After the model returns a hypothesis, that text is embedded as a document (the same input type used for stored fragments) and compared with **every** stored vector, parents included. A paraphrase of a parent must fail.

```
N(h) = 1 - max(0, max_f cos(e_h, e_f))
```

| max cosine | status | what happens |
| --- | --- | --- |
| > `REM_NOVELTY_CEILING` (0.88) | `too_similar` | no critic call |
| < `REM_INCOHERENCE_FLOOR` (0.05) | `incoherent` | no critic call |
| otherwise | `kept` | critic runs |

`no_genuine_link` from the author model also skips the critic. The raw nearest cosine is stored even when it is negative; `N(h)` itself stays in `[0, 1]`.

## Critic

The critic is a second call, with its own temperature (`REM_CRITIC_TEMPERATURE`, default 0). It cannot see the tool result of the author call. The prompt fixes the anchors:

| axis | 0 | 0.5 | 1 |
| --- | --- | --- | --- |
| novelty | paraphrases a parent | minor qualification | not entailed by either parent alone |
| logical coherence | non sequitur or self-contradiction | a gap in the argument | follows from the stated premises |
| falsifiability | no observation named | a vague consequence | an observation that would refute it |
| cross-domain utility | stuck in the source wording | useful next door | a concrete transfer or experiment |

Anthropic calls must use the `submit_hypothesis` or `submit_critique` tool. The composite is the weighted sum

```
C = 0.30 N + 0.30 L + 0.20 F + 0.20 U
```

The weights have to sum to 1 or configuration fails. A candidate is accepted only when `C ≥ REM_CRITIC_THRESHOLD` (0.60) **and** every axis is at least `REM_CRITIC_FLOOR` (0.30). A report with axes `(1, 1, 1, 0.2)` has composite 0.84 and is still rejected. Accepted hypotheses are inserted as fragments with source `recombination` and a lineage edge, unless `--no-materialize` is set.

## Reproducibility

`load_config()` reads the variables in [`.env.example`](.env.example). `ExperimentConfig.public_dict()` is what gets stored. It includes model names, both temperatures, the seed, every threshold, the critic weights, and the strategy. It does not include `ANTHROPIC_API_KEY` or `VOYAGE_API_KEY`.

What is reproducible offline, given the same fragments and config: cluster assignment, pair ranking, the novelty decision, and the accept bit for a given score vector. What is not bitwise reproducible even at temperature 0: the provider's text. Pin `ANTHROPIC_MODEL` and `VOYAGE_EMBED_MODEL` anyway. Do not mix embedders or widths in one database; the store rejects a vector whose length differs from the rows already present.

`voyage-3.5` returns **1024** floats by default and also supports 256, 512, and 2048 (`REM_EMBED_DIM`). The width is whatever Voyage documents for that model, and each row records the width that was actually stored. See the [Voyage embeddings reference](https://docs.voyageai.com/docs/embeddings).

The hashing embedder (`--embedder hashing`) is a signed feature hash into 32 dimensions. It exists so the pipeline runs without keys. It is not a semantic space. Do not interpret its cosines as evidence.

## Setup

```bash
cd rem-recombinator
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in the two keys; .env stays untracked
```

Personal exports and runtime databases are gitignored (`*.db`, `memories/*.json`). Keep them on disk. Do not force-add them.

## CLI

```bash
python -m rem_recombinator import examples/sample_memory_export.json --dry-run
python -m rem_recombinator import examples/sample_memory_export.json --embedder hashing
python -m rem_recombinator run --embedder hashing --llm template --strategy all --n 3 --seed 0
python -m rem_recombinator stats
python -m rem_recombinator export --format markdown --output run.md
python -m rem_recombinator lineage 1
python -m rem_recombinator diagnose --seed 0
python -m rem_recombinator evaluate examples/evaluation_labels.example.json
```

`python cli.py …` is the same entry point. A live run is:

```bash
python -m rem_recombinator import /path/to/memories-export.json --embedder voyage
python -m rem_recombinator run --strategy semantic_bridge --n 5 --seed 0
```

`--llm template` is the offline stand-in. On the sample export it keeps two candidates and accepts none, because every template axis is 0.5 and the composite 0.50 is below 0.60. That is the gate working, not a successful scientific recombination. `--verbose` prints the pair-selection log on stderr.

Extraction from a Claude memories export:

- each `- [stated]` line in `memory_files` becomes a fragment; the domain is the first directory of the path (`/physics/thermo.md` → `physics`);
- `project_memories` split on blank lines, domain `project`;
- `conversations_memory` split the same way, domain `conversation`;
- paragraphs shorter than 20 characters are dropped;
- text already stored is skipped, so a re-import resumes.

## MCP

`python -m rem_recombinator.mcp_server` speaks MCP over stdio. The module does not call `print`. Logs are attached to stderr inside `serve()`.

Tools:

| tool | arguments |
| --- | --- |
| `run_rem_cycle` | `strategy` (`semantic_bridge`, `analogical_transfer`, `dialectic_synthesis`, `all`), `candidates` (1–20), `seed`, `materialize` |
| `list_accepted_insights` | `limit` (1–100) |
| `store_stats` | none |
| `fragment_lineage` | `fragment_id` |

```json
{
  "mcpServers": {
    "rem-recombinator": {
      "command": "python",
      "args": ["-m", "rem_recombinator.mcp_server"],
      "cwd": "/absolute/path/to/rem-recombinator",
      "env": {
        "ANTHROPIC_API_KEY": "...",
        "VOYAGE_API_KEY": "..."
      }
    }
  }
}
```

Unknown fields, booleans passed where an integer is required, and strategy names outside the enum are rejected before a model call.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite covers store migration, cosine edge cases, all three selectors, critic parsing (including fenced JSON and the dimension floor), the novelty gate, lineage, redaction of `sk-ant-…` in stored errors, seed import from JSON and zip, the threshold report, the CLI, and the MCP handlers. Embeddings and model payloads are fixtures. No network.

## Limits, and what to measure next

- Leader clustering is greedy and id-ordered. A spherical k-means (or any method with a stability check across seeds) is the replacement once the corpus is large enough that the enumeration cap starts sampling.
- Opposition cues miss paraphrased contradictions and fire on accidental word matches. A next experiment is a small hand-labeled set of genuine conflicts, then a comparison of cue precision against a model that only labels “conflict / no conflict”.
- The 0.88 / 0.05 / 0.35 thresholds are still declared defaults. `evaluate` scores a label file (bridge, analogy, dialectic, novelty status, critic axes, midpoint hold-out) and prints the best F1 on that file. It does not write a new default. Publish the table with the labels; change `REM_*` only in a later commit that cites that table.
- Critic scores need a human inter-rater study before they are treated as measurements. The report's MAE is only as good as the gold file.
- The midpoint hold-out checks a geometric claim: the fragment you withheld should be the nearest neighbour of the parent midpoint. It does not show that an accepted sentence predicts unseen text. That still needs a reader.
- The store interface is the seam for pgvector. Nothing in `recombinator.py` depends on SQLite beyond the `Store` methods.
- Provider text at temperature 0 can still change between deploys. Store the model id, which this already does, and treat a model upgrade as a new experiment, not a continuation.

An earlier commit on `main` (`698efc0`) added `data/fragments.db` and a personal memory export. Those paths are no longer tracked. They remain in that commit's tree until history is rewritten.
