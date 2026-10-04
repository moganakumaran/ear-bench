#!/usr/bin/env python3
"""Characterize the FDABench labelled subset that this paper builds on.

Writes results/characterization.json. Every number the manuscript quotes about
the base benchmark comes from this file; nothing is transcribed by hand.
Normalisation is shared with build_benchmark.py via common.py.

Run:  ./.venv/bin/python characterize.py
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np

from common import (
    GOVERNED_TOOLS,
    RESULTS,
    TOOL_ALIASES,
    config_totals,
    load_labelled,
)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    df = load_labelled()
    totals = config_totals()
    n = len(df)
    n_all = sum(v["total"] for v in totals.values())

    aliased = sum(
        1
        for route in df["gold_route"]
        if any(s.get("tool") in TOOL_ALIASES for s in route)
    )

    freq = Counter()
    for tool_set in df["gold_tool_set"]:
        freq.update(tool_set)

    report = {
        "source": "FDAbench2026/FDAbench-Full (HuggingFace parquet export)",
        "base_paper": "FDABench, arXiv:2509.02473, KDD'26",
        "tasks_total": n_all,
        "tasks_labelled": n,
        "labelled_fraction": round(n / n_all, 4),
        "per_config": totals,
        "by_level": df["level"].value_counts().to_dict(),
        "by_database_type": df["database_type"].value_counts().to_dict(),
        "by_question_type": df["question_type"].value_counts().to_dict(),
        "distinct_databases": int(df["gold_db"].nunique()),
        "route_length": {
            "mean": round(float(np.mean([len(t) for t in df["gold_tools"]])), 2),
            "min": int(min(len(t) for t in df["gold_tools"])),
            "max": int(max(len(t) for t in df["gold_tools"])),
        },
        "tasks_with_depends_on": int(
            sum(1 for r in df["gold_route"] if any("depends_on" in s for s in r))
        ),
        "sql_alias_normalised_tasks": aliased,
        "tasks_with_db_column_disagreement": int(df["db_column_disagrees"].sum()),
        "tool_frequency": {
            tool: {"tasks": cnt, "pct": round(100 * cnt / n, 1)}
            for tool, cnt in freq.most_common()
        },
        "distinct_gold_tool_sets": int(df["gold_tool_set"].nunique()),
        "gold_tool_sets": [
            {"tools": list(s), "tasks": c}
            for s, c in Counter(df["gold_tool_set"]).most_common()
        ],
        "tool_set_size_distribution": {
            str(k): v
            for k, v in sorted(Counter(len(s) for s in df["gold_tool_set"]).items())
        },
        "governed_tool_coverage": {
            tool: {"tasks": freq.get(tool, 0), "pct": round(100 * freq.get(tool, 0) / n, 1)}
            for tool in GOVERNED_TOOLS
        },
        "database_task_counts": df["gold_db"].value_counts().to_dict(),
    }

    path = RESULTS / "characterization.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=False))
    print(f"wrote {path}")
    print(f"  labelled tasks        : {n} of {n_all}")
    print(f"  distinct databases    : {report['distinct_databases']}")
    print(f"  distinct gold routes  : {report['distinct_gold_tool_sets']}")
    print(f"  sql alias normalised  : {aliased} tasks")
    print(f"  db-column disagreement: {report['tasks_with_db_column_disagreement']} tasks")


if __name__ == "__main__":
    main()
