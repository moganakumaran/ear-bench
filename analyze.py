#!/usr/bin/env python3
"""Turn router decisions into the paper's numbers.

Reports every rate with a Wilson interval, and compares routers with exact
McNemar on the same instances, Holm-adjusted across the comparison family.

The headline comparison is R6 vs R4: R4 is the honest strong baseline — an LLM
handed the entitlement list in its prompt — so beating R1/R2/R3/R5 proves very
little on its own.

Run:  ./.venv/bin/python analyze.py [--tag haiku]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

from common import RESULTS
from stats import holm, mcnemar, rate_ci

PRIMARY = "R6_entitlement_aware"
BASELINE = "R4_llm_entitled"
STRATA = ("FEASIBLE", "INFEASIBLE_SCOPE", "INFEASIBLE_TOOL")
CONDITIONS = ("A_unrestricted", "B_natural", "C_foreign")
RATES = ("over_reach", "safe", "useful", "exact_route")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="haiku")
    args = ap.parse_args()

    path = RESULTS / f"decisions_{args.tag}.jsonl"
    rows = [json.loads(line) for line in path.open()]
    by_router: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_router[r["router"]].append(r)
    routers = sorted(by_router)

    out: dict = {"tag": args.tag, "routers": {}, "comparisons": {}}

    for name in routers:
        rrows = by_router[name]
        entry = {
            "n": len(rrows),
            "overall": {k: rate_ci(rrows, k).as_dict() for k in RATES},
            "abstention": {
                "precision": rate_ci(
                    [r for r in rrows if r["abstained"]], "should_abstain"
                ).as_dict(),
                "recall": rate_ci(
                    [r for r in rrows if r["should_abstain"]], "abstained"
                ).as_dict(),
            },
        }
        for cond in CONDITIONS:
            sub = [r for r in rrows if r["condition"] == cond]
            entry[cond] = {k: rate_ci(sub, k).as_dict() for k in RATES}
        for st in STRATA:
            sub = [r for r in rrows if r["stratum"] == st]
            entry[st] = {k: rate_ci(sub, k).as_dict() for k in RATES}
        out["routers"][name] = entry

    # --- paired comparisons against the strong baseline -------------------
    raw_p: dict[str, float] = {}
    for name in routers:
        if name == BASELINE:
            continue
        for key in ("over_reach", "safe", "useful"):
            res = mcnemar(by_router[name], by_router[BASELINE], key)
            label = f"{name}_vs_{BASELINE}__{key}"
            out["comparisons"][label] = res
            if "p_value" in res:
                raw_p[label] = res["p_value"]
    adjusted = holm(raw_p)
    for label, p in adjusted.items():
        out["comparisons"][label]["p_holm"] = p

    (RESULTS / f"analysis_{args.tag}.json").write_text(json.dumps(out, indent=2))

    # --- console report ---------------------------------------------------
    print(f"analysis for tag={args.tag}  ({len(rows)} decisions)\n")
    hdr = f"{'router':24s} {'over-reach %':>22s} {'safe %':>22s} {'useful %':>22s}"
    print(hdr)
    print("-" * len(hdr))
    for name in routers:
        e = out["routers"][name]["overall"]
        def fmt(d):
            return f"{100*d['point']:5.1f} [{100*d['ci_low']:4.1f},{100*d['ci_high']:5.1f}]"
        print(f"{name:24s} {fmt(e['over_reach']):>22s} {fmt(e['safe']):>22s} {fmt(e['useful']):>22s}")

    print("\nover-reach by condition (point estimates)")
    print(f"  {'router':24s}" + "".join(f"{c:>18s}" for c in CONDITIONS))
    for name in routers:
        e = out["routers"][name]
        print(
            f"  {name:24s}"
            + "".join(f"{100*e[c]['over_reach']['point']:>17.1f}%" for c in CONDITIONS)
        )

    print(f"\npaired McNemar vs {BASELINE} (Holm-adjusted)")
    for label in sorted(out["comparisons"]):
        res = out["comparisons"][label]
        if "p_value" not in res:
            continue
        router, key = label.split("__")
        print(
            f"  {router.replace('_vs_'+BASELINE,''):24s} {key:11s} "
            f"a_only={res['a_only']:4d} b_only={res['b_only']:4d} "
            f"p={res['p_value']:.3e} holm={res.get('p_holm'):.3e}"
        )

    print(f"\nwrote {RESULTS}/analysis_{args.tag}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
