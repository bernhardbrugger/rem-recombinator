# REM Recombinator

Framework for reproducible semantic recombination of a personal knowledge store. The full manual, the worked numerical example, and the command reference live in [`rem-recombinator/README.md`](rem-recombinator/README.md). The package itself is [`rem-recombinator/`](rem-recombinator/).

The name refers to a computational sketch of REM sleep, not to a claim about brains. Distant fragments are reactivated together, a model proposes one hypothesis, and a separate critic keeps almost none of them. Pair selection, the novelty gate, and the accept bit are deterministic functions of the stored vectors and the saved hyperparameters.

```
export -> embed -> SQLite fragments
                      |
                      +-- semantic bridge | analogical transfer | dialectic
                      |
                      v
                    hypothesis -> novelty filter -> four-axis critic -> lineage
```

Cosine similarity is `cos(u, v) = (u · v) / (||u|| ||v||)`. A hypothesis is dropped when its nearest neighbour is closer than 0.88 or farther than 0.05. The critic scores novelty, logical coherence, falsifiability, and cross-domain utility in `[0, 1]`. The code accepts a report only when the weighted composite is at least 0.60 and every axis is at least 0.30.

```bash
cd rem-recombinator
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
pytest
python -m rem_recombinator import examples/sample_memory_export.json --embedder hashing
python -m rem_recombinator evaluate examples/evaluation_labels.example.json
```

`evaluate` reports precision and recall for the similarity thresholds. It does not change the defaults.

`.env`, `*.db`, and `memories/*.json` are gitignored. API keys stay local.
