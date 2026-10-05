# EAR-Bench — Entitlement-Aware Routing benchmark

Artefact for the paper **"When Policy Contradicts Content: Prompted Entitlement
Compliance in LLM Agent Routing"** (Ubi-Media 2027, Springer CCIS).

Enterprise agent routers are normally evaluated without any notion of *who is
asking*. This artefact adds a principal and entitlement layer to a public
data-agent benchmark, defines **over-reach** — the fraction of emitted routes
that touch a tool or database the principal may not use — and measures it for
seven routers.

## What is here

| | |
|---|---|
| `entitlements.yaml` | The declared policy: 8 data domains over 73 databases, 3 governed tools, 7 principals. Hand-written, version-controlled, asserted complete at build time. |
| `build_benchmark.py` | Builds the entitlement-extended benchmark. `--policy-seed N` builds a shuffled-policy arm; `--opaque-principals` builds the one-factor name control. |
| `build_pairs.py` | Paired permit/revoke construction: toggles one entitlement the gold route requires, holding query, task, route, source and principal fixed. Build-time gates abort on a malformed pair. |
| `validate_benchmark.py` | 23 independent checks that re-derive expectations from the policy and the raw data rather than trusting the builder. |
| `routers.py`, `run_routers.py` | The seven routers and the experiment orchestrator. |
| `metrics.py`, `stats.py` | Scoring and statistics, both with self-tests. Written before any router was run. |
| `analyze.py`, `analyze_pairs.py`, `ablations.py`, `cross_arm.py`, `filtered_baselines.py`, `error_analysis.py`, `policy_sensitivity.py`, `reference_bounds.py` | Analyses reported in the paper. |
| `results/` | Every generated result file behind every number in the paper. |

## Base data

The underlying tasks come from **FDABench** (arXiv:2509.02473, KDD 2026, MIT),
dataset `FDAbench2026/FDAbench-Full` on HuggingFace. It is **re-fetched, never
redistributed**:

```bash
python -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/python fetch_data.py        # downloads the parquet shards
```

Of its 2,007 tasks, **834 carry gold routes**; the rest are a withheld
leaderboard split. Three defects in the base data are normalised and documented
in `common.py` rather than silently repaired.

## Reproducing

```bash
./.venv/bin/python characterize.py          # base-benchmark characterisation
./.venv/bin/python build_benchmark.py       # declared policy
./.venv/bin/python build_benchmark.py --opaque-principals
./.venv/bin/python build_benchmark.py --policy-seed 0   # and 1, 2
./.venv/bin/python validate_benchmark.py    # expect 23/23
./.venv/bin/python reference_bounds.py      # construction bounds, no model needed
```

Router runs need an Anthropic key in `ANTHROPIC_API_KEY` or a local `.env`:

```bash
./.venv/bin/python run_routers.py --seeds 1 --temperature 0.0
./.venv/bin/python analyze.py --tag haiku
```

Every model response is cached on disk by prompt hash, so a re-run of an
unchanged experiment costs nothing and an interrupted run resumes for free. The
full set of experiments in the paper cost **US$17.07** across roughly 11,300
calls.

`results/` is committed in full, including the per-instance decision files, so
every analysis script can be re-run and every number in the paper re-derived
**without making a single model call**. That is also why the repository is ~80 MB.

## Licence

Code and the entitlement layer: MIT (see `LICENSE`). The underlying FDABench
tasks remain under their own licence and are not redistributed here.
