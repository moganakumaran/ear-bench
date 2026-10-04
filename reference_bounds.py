#!/usr/bin/env python3
"""Degenerate and oracle reference bounds for the entitlement benchmark.

These need no model and no API key. They frame every router result:

  R0a always-abstain          perfectly safe, perfectly useless. Shows why
                              safe_rate alone is gameable and useful_rate exists.
  R0b permission-blind oracle PERFECT routing knowledge, zero entitlement
                              awareness. The ceiling for any permission-blind
                              router, and the number the paper argues against.
  R0c entitlement-aware oracle the attainable ceiling.

R0b over-reaches exactly on the instances that should have abstained. That is
structural, not empirical: it always runs the full gold route, which breaches
entitlement precisely when some element of that route is unentitled.

Run:  ./.venv/bin/python reference_bounds.py
"""
from __future__ import annotations

import json

from common import RESULTS
from metrics import RouteDecision, aggregate, score_instance

CONDITIONS = ("A_unrestricted", "B_natural", "C_foreign")


def always_abstain(_inst: dict) -> RouteDecision:
    return RouteDecision.abstain("degenerate")


def blind_oracle(inst: dict) -> RouteDecision:
    """True gold route every time; entitlements never consulted."""
    tools = inst["gold_tools"]
    return RouteDecision(tools=frozenset(tools), database=inst["gold_db"])


def aware_oracle(inst: dict) -> RouteDecision:
    if inst["stratum"] != "FEASIBLE":
        return RouteDecision.abstain("not entitled")
    return RouteDecision(
        tools=frozenset(inst["entitled_gold_tools"]), database=inst["entitled_gold_db"]
    )


ROUTERS = {
    "R0a_always_abstain": always_abstain,
    "R0b_permission_blind_oracle": blind_oracle,
    "R0c_entitlement_aware_oracle": aware_oracle,
}


def main() -> None:
    instances = [json.loads(line) for line in (RESULTS / "benchmark.jsonl").open()]
    out = {}
    for name, fn in ROUTERS.items():
        scored = [score_instance(i, fn(i)) for i in instances]
        entry = {"overall": aggregate(scored)}
        for cond in CONDITIONS:
            entry[cond] = aggregate([s for s in scored if s["condition"] == cond])
        for stratum in ("FEASIBLE", "INFEASIBLE_SCOPE", "INFEASIBLE_TOOL"):
            entry[stratum] = aggregate([s for s in scored if s["stratum"] == stratum])
        out[name] = entry

    path = RESULTS / "reference_bounds.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}\n")
    hdr = f"{'router':32s} {'over_reach':>11s} {'safe':>7s} {'useful':>7s}"
    print(hdr)
    print("-" * len(hdr))
    for name, entry in out.items():
        o = entry["overall"]
        print(
            f"{name:32s} {o['over_reach_rate']:>11.4f} "
            f"{o['safe_rate']:>7.4f} {o['useful_rate']:>7.4f}"
        )
    blind = out["R0b_permission_blind_oracle"]
    print(
        "\nR0b over-reach by condition: "
        + ", ".join(f"{c}={blind[c]['over_reach_rate']:.4f}" for c in CONDITIONS)
    )


if __name__ == "__main__":
    main()
