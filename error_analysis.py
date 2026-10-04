#!/usr/bin/env python3
"""Where each router fails, and why. Zero API calls.

Three questions the results raise:
  1. R4 is handed the entitlement list. Where does it still over-reach?
  2. What does R6 give up for its zero over-reach?
  3. Do the two routers fail in the same places?

Run:  ./.venv/bin/python error_analysis.py [--tag haiku]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

from common import RESULTS
from routers import GOVERNED, SQL_CORE

PRIMARY = "R6_entitlement_aware"
BASELINE = "R4_llm_entitled"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="haiku")
    args = ap.parse_args()

    rows = [json.loads(l) for l in (RESULTS / f"decisions_{args.tag}.jsonl").open()]
    bench = {
        json.loads(l)["instance_id"]: json.loads(l)
        for l in (RESULTS / "benchmark.jsonl").open()
    }
    by_router = defaultdict(list)
    for r in rows:
        by_router[r["router"]].append(r)

    out: dict = {"tag": args.tag}

    for name in (BASELINE, PRIMARY):
        rrows = by_router[name]
        feasible = [r for r in rrows if r["stratum"] == "FEASIBLE"]
        over = [r for r in rrows if r["over_reach"]]

        unheld = Counter()
        for r in over:
            b = bench[r["instance_id"]]
            for t in set(r["tools"]) - set(SQL_CORE):
                if t not in b["entitled_governed_tools"]:
                    unheld[t] += 1

        out[name] = {
            "over_reach": {
                "n": len(over),
                "by_condition": dict(Counter(r["condition"] for r in over)),
                "by_stratum": dict(Counter(r["stratum"] for r in over)),
                "by_principal": dict(Counter(r["principal"] for r in over)),
                "unheld_tool_invoked": dict(unheld),
            },
            "false_abstention": {
                "n": len([r for r in feasible if r["abstained"]]),
                "of_feasible": len(feasible),
                "by_principal": dict(
                    Counter(r["principal"] for r in feasible if r["abstained"])
                ),
                "top_reasons": Counter(
                    r["rationale"][:60] for r in feasible if r["abstained"]
                ).most_common(5),
            },
            "missed_abstention": {
                "n": len([r for r in rrows if r["should_abstain"] and not r["abstained"]]),
                "by_stratum": dict(
                    Counter(
                        r["stratum"]
                        for r in rrows
                        if r["should_abstain"] and not r["abstained"]
                    )
                ),
            },
        }

    # Do the two routers fail together or independently?
    a = {r["instance_id"]: r for r in by_router[PRIMARY]}
    b = {r["instance_id"]: r for r in by_router[BASELINE]}
    shared = sorted(set(a) & set(b))
    joint = Counter()
    for iid in shared:
        joint[(bool(a[iid]["safe"]), bool(b[iid]["safe"]))] += 1
    out["joint_safety"] = {
        "both_safe": joint[(True, True)],
        "only_R6_safe": joint[(True, False)],
        "only_R4_safe": joint[(False, True)],
        "neither_safe": joint[(False, False)],
    }

    (RESULTS / f"error_analysis_{args.tag}.json").write_text(json.dumps(out, indent=2, default=str))

    print(f"error analysis ({args.tag})\n")
    for name in (BASELINE, PRIMARY):
        e = out[name]
        print(f"{name}")
        o = e["over_reach"]
        print(f"  over-reach           : {o['n']:4d}   strata={o['by_stratum']}")
        if o["unheld_tool_invoked"]:
            print(f"    unheld tool invoked: {o['unheld_tool_invoked']}")
            print(f"    concentrated in    : {o['by_principal']}")
        f = e["false_abstention"]
        print(f"  false abstention     : {f['n']:4d} of {f['of_feasible']} feasible   {f['by_principal']}")
        m = e["missed_abstention"]
        print(f"  missed abstention    : {m['n']:4d}   strata={m['by_stratum']}\n")
    print("joint safety:", out["joint_safety"])
    print(f"\nwrote {RESULTS}/error_analysis_{args.tag}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
