#!/usr/bin/env python3
"""Is a router reading the entitlement list, or guessing the domain? Zero API calls.

In the declared policy a database's domain is semantically predictable from the
query: a COVID question obviously belongs to health_bio. A router can therefore
look entitlement-aware while really inferring the domain from query content. The
shuffled arm reassigns databases to domains at random, severing that cue.

All comparisons here are CONDITIONAL ON STRATUM. The two arms have different
strata sizes, so headline rates are not comparable across arms — a router whose
behaviour cannot depend on the policy at all (R2) still shows a large apparent
shift in its unconditional useful_rate purely from composition.

Run:  ./.venv/bin/python policy_sensitivity.py
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

from common import RESULTS
from stats import wilson

STRATA = ("INFEASIBLE_SCOPE", "INFEASIBLE_TOOL", "FEASIBLE")


def load(tag: str) -> list[dict]:
    return [json.loads(l) for l in (RESULTS / f"decisions_{tag}.jsonl").open()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--declared", default="haiku")
    ap.add_argument("--shuffled", default="haiku_shuffled_0")
    ap.add_argument(
        "--routers", nargs="*", default=["R4_llm_entitled", "R6_entitlement_aware"]
    )
    args = ap.parse_args()

    arms = {"declared": load(args.declared), "shuffled": load(args.shuffled)}
    out: dict = {"arms": {k: args.declared if k == "declared" else args.shuffled for k in arms}}

    ref = [r for r in arms["declared"] if r["router"] == args.routers[0]]
    out["strata_sizes"] = {
        name: dict(
            Counter(
                r["stratum"] for r in rows if r["router"] == args.routers[0]
            )
        )
        for name, rows in arms.items()
    }

    out["over_reach_by_stratum"] = {}
    for router in args.routers:
        out["over_reach_by_stratum"][router] = {}
        for st in STRATA:
            entry = {}
            for name, rows in arms.items():
                sub = [r for r in rows if r["router"] == router and r["stratum"] == st]
                entry[name] = wilson(
                    sum(bool(r["over_reach"]) for r in sub), len(sub)
                ).as_dict()
            out["over_reach_by_stratum"][router][st] = entry

    out["false_abstention_on_feasible"] = {}
    out["abstention_precision"] = {}
    for router in args.routers:
        fa, ap_ = {}, {}
        for name, rows in arms.items():
            feas = [r for r in rows if r["router"] == router and r["stratum"] == "FEASIBLE"]
            fa[name] = wilson(sum(bool(r["abstained"]) for r in feas), len(feas)).as_dict()
            ab = [r for r in rows if r["router"] == router and r["abstained"]]
            ap_[name] = wilson(sum(bool(r["should_abstain"]) for r in ab), len(ab)).as_dict()
        out["false_abstention_on_feasible"][router] = fa
        out["abstention_precision"][router] = ap_

    (RESULTS / "policy_sensitivity.json").write_text(json.dumps(out, indent=2))

    def fmt(d):
        return f"{100*d['point']:5.1f} [{100*d['ci_low']:4.1f},{100*d['ci_high']:5.1f}] n={d['n']}"

    print("policy sensitivity: declared vs shuffled, conditional on stratum\n")
    print("strata sizes:", out["strata_sizes"], "\n")
    print(f"{'router':22s} {'stratum':18s} {'declared':>22s} {'shuffled':>22s}")
    print("-" * 86)
    for router in args.routers:
        for st in STRATA:
            e = out["over_reach_by_stratum"][router][st]
            print(f"{router:22s} {st:18s} {fmt(e['declared']):>22s} {fmt(e['shuffled']):>22s}")
    print("\nfalse abstention on FEASIBLE")
    for router in args.routers:
        e = out["false_abstention_on_feasible"][router]
        print(f"  {router:22s} {fmt(e['declared']):>22s} {fmt(e['shuffled']):>22s}")
    print("\nabstention precision")
    for router in args.routers:
        e = out["abstention_precision"][router]
        print(f"  {router:22s} {fmt(e['declared']):>22s} {fmt(e['shuffled']):>22s}")
    print(f"\nwrote {RESULTS}/policy_sensitivity.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
