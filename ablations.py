#!/usr/bin/env python3
"""Ablations of the proposed router. Zero API calls.

R6's model call is a principal-independent extraction of what the query needs.
Every ablation changes only what happens AFTER that call, so each one re-scores
the SAME cached extractions. That is why the ablation grid is free, and it also
means the ablations are exactly comparable — no decode noise between arms.

Arms
----
full              the proposed router
no_entitlement    drop the hard data-scope filter; keep the tool feasibility
                  test. Isolates how much of the result is data-scope checking.
no_feasibility    keep the data-scope filter; act with whatever tools are held
                  instead of abstaining when a required tool is missing. This is
                  the "degrade silently" policy.
no_extraction     ignore the model's reading of the query and always request
                  every governed tool the principal holds. Isolates the value of
                  the extraction call.

Run:  ./.venv/bin/python ablations.py [--tag haiku]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

from common import RESULTS
from metrics import RouteDecision, aggregate, score_instance
from routers import GOVERNED, SQL_CORE
from stats import holm, mcnemar, rate_ci


def needs_from(need: dict | None) -> set[str]:
    if not need:
        return set()
    return {t for t in GOVERNED if need.get(f"needs_{t}") is True}


def arm_full(need, inst) -> RouteDecision:
    if not inst["database_entitled"]:
        return RouteDecision.abstain("out of data scope")
    missing = needs_from(need) - set(inst["entitled_governed_tools"])
    if missing:
        return RouteDecision.abstain(f"unheld: {sorted(missing)}")
    return RouteDecision(
        tools=frozenset(SQL_CORE | needs_from(need)), database=inst["gold_db"]
    )


def arm_no_entitlement(need, inst) -> RouteDecision:
    """No data-scope filter. Tool feasibility still applies."""
    missing = needs_from(need) - set(inst["entitled_governed_tools"])
    if missing:
        return RouteDecision.abstain(f"unheld: {sorted(missing)}")
    return RouteDecision(
        tools=frozenset(SQL_CORE | needs_from(need)), database=inst["gold_db"]
    )


def arm_no_feasibility(need, inst) -> RouteDecision:
    """Degrade silently: proceed with only the tools the principal holds."""
    if not inst["database_entitled"]:
        return RouteDecision.abstain("out of data scope")
    usable = needs_from(need) & set(inst["entitled_governed_tools"])
    return RouteDecision(tools=frozenset(SQL_CORE | usable), database=inst["gold_db"])


def arm_no_extraction(need, inst) -> RouteDecision:
    """Ignore the query; request everything the principal is allowed to use."""
    if not inst["database_entitled"]:
        return RouteDecision.abstain("out of data scope")
    return RouteDecision(
        tools=frozenset(SQL_CORE | set(inst["entitled_governed_tools"])),
        database=inst["gold_db"],
    )


ARMS = {
    "full": arm_full,
    "no_entitlement": arm_no_entitlement,
    "no_feasibility": arm_no_feasibility,
    "no_extraction": arm_no_extraction,
}


def load_extractions(tag: str) -> dict[str, dict | None]:
    """Recover each task's cached extraction from the recorded R6 decisions.

    The decision rationale encodes what R6 concluded, but the raw extraction is
    what the ablations need, so re-derive it from the tool set R6 emitted on the
    UNRESTRICTED arm, where no filtering was applied.
    """
    rows = [json.loads(l) for l in (RESULTS / f"decisions_{tag}.jsonl").open()]
    need: dict[str, dict | None] = {}
    for r in rows:
        if r["router"] != "R6_entitlement_aware" or r["condition"] != "A_unrestricted":
            continue
        task_id = r["instance_id"].split("::", 1)[0]
        if r["decision_abstained"]:
            need[task_id] = None
            continue
        picked = set(r["tools"]) - set(SQL_CORE)
        need[task_id] = {f"needs_{t}": (t in picked) for t in GOVERNED}
    return need


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="haiku")
    args = ap.parse_args()

    rows = [json.loads(l) for l in (RESULTS / f"decisions_{args.tag}.jsonl").open()]
    instances = {}
    for r in rows:
        instances.setdefault(r["instance_id"], r)
    bench = {
        json.loads(l)["instance_id"]: json.loads(l)
        for l in (RESULTS / "benchmark.jsonl").open()
    }
    need = load_extractions(args.tag)
    eval_ids = [i for i in instances if i in bench]

    out: dict = {"tag": args.tag, "n_instances": len(eval_ids), "arms": {}}
    scored_by_arm: dict[str, list[dict]] = {}
    for name, fn in ARMS.items():
        scored = []
        for iid in eval_ids:
            inst = bench[iid]
            scored.append(score_instance(inst, fn(need.get(inst["task_id"]), inst)))
        scored_by_arm[name] = scored
        agg = aggregate(scored)
        out["arms"][name] = {
            "aggregate": agg,
            "over_reach": rate_ci(scored, "over_reach").as_dict(),
            "safe": rate_ci(scored, "safe").as_dict(),
            "useful": rate_ci(scored, "useful").as_dict(),
        }

    # Paired tests against the full method. The manuscript previously said
    # "usefulness statistically unchanged" with no test behind it.
    raw: dict[str, float] = {}
    out["vs_full"] = {}
    for name in ARMS:
        if name == "full":
            continue
        out["vs_full"][name] = {}
        for key in ("over_reach", "safe", "useful"):
            res = mcnemar(scored_by_arm[name], scored_by_arm["full"], key)
            out["vs_full"][name][key] = res
            if "p_value" in res:
                raw[f"{name}__{key}"] = res["p_value"]
    for label, p_adj in holm(raw).items():
        arm, key = label.split("__")
        out["vs_full"][arm][key]["p_holm"] = p_adj

    (RESULTS / f"ablations_{args.tag}.json").write_text(json.dumps(out, indent=2))

    hdr = f"{'arm':18s} {'over-reach %':>20s} {'safe %':>20s} {'useful %':>20s}"
    print(f"ablations ({len(eval_ids)} instances, 0 API calls)\n")
    print(hdr)
    print("-" * len(hdr))
    for name in ARMS:
        e = out["arms"][name]
        def f(d):
            return f"{100*d['point']:5.1f} [{100*d['ci_low']:4.1f},{100*d['ci_high']:5.1f}]"
        print(f"{name:18s} {f(e['over_reach']):>20s} {f(e['safe']):>20s} {f(e['useful']):>20s}")
    print("\npaired McNemar vs full (Holm-adjusted within this family)")
    for name, keys in out["vs_full"].items():
        for key, r in keys.items():
            if "p_value" not in r:
                continue
            print(f"  {name:16s} {key:11s} arm_only={r['a_only']:4d} full_only={r['b_only']:4d} "
                  f"p={r['p_value']:.3e} holm={r.get('p_holm', float('nan')):.3e}")
    print(f"\nwrote {RESULTS}/ablations_{args.tag}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
