#!/usr/bin/env python3
"""Does the extraction prompt matter, or does only the filter matter? Zero API calls.

R6 differs from the baselines in two ways at once: it runs a dedicated
principal-independent extraction, AND it applies a hard entitlement filter. A
reviewer will reasonably ask whether the filter alone accounts for the result.

This separates them by applying the SAME post-hoc filter to the cached outputs of
the permission-blind router (R3) and the entitlement-prompted router (R4):

  drop any governed tool the principal does not hold;
  abstain outright when the database is outside the principal's data scope.

If R3+filter matches R6, the contribution is entitlement resolution in general and
the dedicated extraction earns nothing. If it falls short, the extraction earns
its call. Both outcomes are reported as found.

Run:  ./.venv/bin/python filtered_baselines.py [--tag haiku]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

from common import RESULTS
from metrics import RouteDecision, aggregate, score_instance
from routers import SQL_CORE
from stats import holm, mcnemar, rate_ci

SOURCES = {
    "R3_llm_blind": "R3+filter",
    "R4_llm_entitled": "R4+filter",
}
REFERENCE = "R6_entitlement_aware"


def apply_filter(row: dict, inst: dict) -> RouteDecision:
    """The post-hoc entitlement filter, identical for every source router."""
    if not inst["database_entitled"]:
        return RouteDecision.abstain("filtered: database out of data scope")
    if row["decision_abstained"]:
        return RouteDecision.abstain("source router abstained")
    held = set(inst["entitled_governed_tools"])
    kept = (set(row["tools"]) & held) | (set(row["tools"]) & set(SQL_CORE))
    return RouteDecision(
        tools=frozenset(kept | set(SQL_CORE)),
        database=inst["gold_db"],
        rationale="filtered",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="haiku")
    args = ap.parse_args()

    rows = [json.loads(l) for l in (RESULTS / f"decisions_{args.tag}.jsonl").open()]
    bench = {
        json.loads(l)["instance_id"]: json.loads(l)
        for l in (RESULTS / "benchmark.jsonl").open()
    }
    by_router: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_router[r["router"]].append(r)

    scored_by: dict[str, list[dict]] = {}
    for src, label in SOURCES.items():
        out = []
        for r in by_router[src]:
            inst = bench[r["instance_id"]]
            s = score_instance(inst, apply_filter(r, inst))
            s["instance_id"] = r["instance_id"]
            out.append(s)
        scored_by[label] = out

    # Unfiltered sources and the proposed router, for side-by-side reporting.
    for name in (*SOURCES, REFERENCE):
        scored_by[name] = by_router[name]

    report: dict = {"tag": args.tag, "routers": {}, "vs_reference": {}}
    for name, sc in scored_by.items():
        report["routers"][name] = {
            "aggregate": aggregate(sc),
            "over_reach": rate_ci(sc, "over_reach").as_dict(),
            "safe": rate_ci(sc, "safe").as_dict(),
            "useful": rate_ci(sc, "useful").as_dict(),
        }

    raw: dict[str, float] = {}
    for label in SOURCES.values():
        report["vs_reference"][label] = {
            key: mcnemar(scored_by[label], scored_by[REFERENCE], key)
            for key in ("over_reach", "safe", "useful")
        }
        for key, res in report["vs_reference"][label].items():
            if "p_value" in res:
                raw[f"{label}__{key}"] = res["p_value"]
    # Holm across this comparison family. Previously these p-values were raw
    # while the manuscript claimed Holm adjustment throughout.
    for lab, p_adj in holm(raw).items():
        src, key = lab.split("__")
        report["vs_reference"][src][key]["p_holm"] = p_adj

    (RESULTS / f"filtered_baselines_{args.tag}.json").write_text(
        json.dumps(report, indent=2)
    )

    def f(d):
        return f"{100*d['point']:5.1f} [{100*d['ci_low']:4.1f},{100*d['ci_high']:5.1f}]"

    order = ["R3_llm_blind", "R3+filter", "R4_llm_entitled", "R4+filter", REFERENCE]
    hdr = f"{'router':24s} {'over-reach %':>20s} {'safe %':>20s} {'useful %':>20s}"
    print(f"filtered baselines ({args.tag}, 0 API calls)\n")
    print(hdr)
    print("-" * len(hdr))
    for name in order:
        e = report["routers"][name]
        print(f"{name:24s} {f(e['over_reach']):>20s} {f(e['safe']):>20s} {f(e['useful']):>20s}")

    print(f"\npaired McNemar vs {REFERENCE}")
    for label in SOURCES.values():
        for key in ("over_reach", "safe", "useful"):
            r = report["vs_reference"][label][key]
            if "p_value" not in r:
                continue
            note = r.get("note", "")
            print(
                f"  {label:12s} {key:11s} filtered_only={r['a_only']:4d} "
                f"R6_only={r['b_only']:4d} p={r['p_value']:.3e} "
                f"holm={r.get('p_holm', float('nan')):.3e} {note}"
            )
    print(f"\nwrote {RESULTS}/filtered_baselines_{args.tag}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
