#!/usr/bin/env python3
"""Paired permit/revoke construction: the clean policy-semantics conflict control.

Implements EXPERIMENT_DESIGN.md and ROBUSTNESS_DESIGN.md. For one task and one
principal, toggles the grant of a SINGLE entitlement that the task's gold route
actually requires:

    PERMIT   principal holds the entitlement  -> FEASIBLE, act on the gold route
    REVOKE   the same principal does not      -> INFEASIBLE, abstain

Everything else in the pair is identical: query, task id, gold route, data
source, data domain, principal identity, every other grant, and the entitled
DATABASE list (axis T). The query still implies the capability; the policy denies
it. That is a policy-semantics conflict produced by one membership bit.

Axes:
  T  revoke a required governed tool   (primary; also the robustness grid)
  D  revoke the task's data domain     (secondary, for symmetry with scope)

Classification reuses build_benchmark.classify so legality is derived from the
same Eq. (1) as the main benchmark, never re-asserted here.

Run:  ./.venv/bin/python build_pairs.py
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict

import yaml

from build_benchmark import (
    POLICY,
    build_db_index,
    classify,
    entitled_domains,
)
from common import GOVERNED_TOOLS, RESULTS, load_labelled

ABSTAIN = "ABSTAIN"
# Fixed in ROBUSTNESS_DESIGN.md before running. Not revised after seeing results.
TOOL_CONFIGS = [
    ("A_web", ("web_search",)),
    ("B_vector", ("vector_search",)),
    ("C_file", ("file_system",)),
    ("D_web_vector", ("web_search", "vector_search")),
]
MIN_SUPPORT_FOR_STATS = 100


class PairError(RuntimeError):
    """A build-time validity gate failed. Always fatal."""


def eligible_principal(policy, db_index, domain, required, denied) -> str | None:
    """First principal (lexicographic, so reproducible) that can host the pair.

    Must be entitled to the domain, hold every tool to be revoked, and hold every
    OTHER required tool -- otherwise the PERMIT member would already be
    infeasible for an unrelated reason and the pair would not isolate one bit.
    """
    for name in sorted(policy["principals"]):
        if name == "platform_admin":
            continue
        spec = policy["principals"][name]
        if domain not in entitled_domains(policy, name):
            continue
        held = set(spec["governed_tools"])
        if not set(denied) <= held:
            continue
        if not set(required) <= held:
            continue
        return name
    return None


def make_instance(row, policy, domain, principal, tools, domains_ok, arm, pair_id, config):
    """Build one member of a pair, classifying it with the shared rule."""
    held_tools = sorted(tools)
    # classify() takes the policy's view, so pass an overridden principal spec.
    shadow = dict(policy)
    shadow["principals"] = dict(policy["principals"])
    shadow["principals"][principal] = {
        **policy["principals"][principal],
        "governed_tools": held_tools,
        "data_domains": sorted(domains_ok),
    }
    stratum, missing = classify(shadow, principal, domain, row["required_governed"])
    db_ok = stratum != "INFEASIBLE_SCOPE"
    return {
        "instance_id": f"{row['task_id']}::{config}::{arm}",
        "pair_id": pair_id,
        "arm": arm,
        "config": config,
        "task_id": row["task_id"],
        "level": row["level"],
        "question_type": row["question_type"],
        "database_type": row["database_type"],
        "query": row["query"],
        "gold_db": row["gold_db"],
        "data_domain": domain,
        "gold_tools": list(row["gold_tool_set"]),
        "required_governed": list(row["required_governed"]),
        "condition": arm,
        "principal": principal,
        "entitled_domains": sorted(domains_ok),
        "entitled_databases": sorted(
            db for db, dom in build_db_index(policy).items() if dom in domains_ok
        ),
        "entitled_databases_count": sum(
            1 for dom in build_db_index(policy).values() if dom in domains_ok
        ),
        "entitled_governed_tools": held_tools,
        "stratum": stratum,
        "missing_governed_tools": missing,
        "database_entitled": db_ok,
        "entitled_gold_tools": (
            list(row["gold_tool_set"]) if stratum == "FEASIBLE" else ABSTAIN
        ),
        "entitled_gold_db": row["gold_db"] if db_ok else None,
    }


def validate(permit: dict, revoke: dict, axis: str) -> None:
    if permit["stratum"] != "FEASIBLE":
        raise PairError(f"{permit['instance_id']}: PERMIT member is {permit['stratum']}")
    expect = "INFEASIBLE_TOOL" if axis == "T" else "INFEASIBLE_SCOPE"
    if revoke["stratum"] != expect:
        raise PairError(f"{revoke['instance_id']}: REVOKE member is {revoke['stratum']}, want {expect}")
    for f in ("task_id", "query", "gold_db", "data_domain", "principal"):
        if permit[f] != revoke[f]:
            raise PairError(f"{permit['pair_id']}: {f} differs across the pair")
    if permit["gold_tools"] != revoke["gold_tools"]:
        raise PairError(f"{permit['pair_id']}: gold route differs across the pair")
    if axis == "T":
        if permit["entitled_databases"] != revoke["entitled_databases"]:
            raise PairError(f"{permit['pair_id']}: database list differs on a tool-axis pair")
        delta = set(permit["entitled_governed_tools"]) ^ set(revoke["entitled_governed_tools"])
        if not delta:
            raise PairError(f"{permit['pair_id']}: revocation changed nothing")
    else:
        delta = set(permit["entitled_domains"]) ^ set(revoke["entitled_domains"])
        if len(delta) != 1:
            raise PairError(f"{permit['pair_id']}: domain axis changed {len(delta)} domains")


def main() -> int:
    policy = yaml.safe_load(POLICY.read_text())
    db_index = build_db_index(policy)
    df = load_labelled()

    instances: list[dict] = []
    support: Counter = Counter()
    per_principal: dict[str, Counter] = defaultdict(Counter)

    # ---- Axis T: revoke required governed tool(s) -------------------------
    for config, denied in TOOL_CONFIGS:
        for _, row in df.iterrows():
            req = set(row["required_governed"])
            if not set(denied) <= req:
                continue  # the gold route does not require what we would revoke
            domain = db_index[row["gold_db"]]
            principal = eligible_principal(policy, db_index, domain, req, denied)
            if principal is None:
                continue
            held = set(policy["principals"][principal]["governed_tools"])
            doms = entitled_domains(policy, principal)
            pair_id = f"{row['task_id']}::{config}"
            permit = make_instance(row, policy, domain, principal, held, doms,
                                   "PERMIT", pair_id, config)
            revoke = make_instance(row, policy, domain, principal, held - set(denied),
                                   doms, "REVOKE", pair_id, config)
            validate(permit, revoke, "T")
            instances += [permit, revoke]
            support[config] += 1
            per_principal[config][principal] += 1

    # ---- Axis D: revoke the task's data domain ----------------------------
    for _, row in df.iterrows():
        domain = db_index[row["gold_db"]]
        req = set(row["required_governed"])
        principal = eligible_principal(policy, db_index, domain, req, ())
        if principal is None:
            continue
        held = set(policy["principals"][principal]["governed_tools"])
        doms = entitled_domains(policy, principal)
        if len(doms) < 2:
            continue  # revoking the only domain leaves an empty scope
        pair_id = f"{row['task_id']}::D_domain"
        permit = make_instance(row, policy, domain, principal, held, doms,
                               "PERMIT", pair_id, "D_domain")
        revoke = make_instance(row, policy, domain, principal, held, doms - {domain},
                               "REVOKE", pair_id, "D_domain")
        validate(permit, revoke, "D")
        instances += [permit, revoke]
        support["D_domain"] += 1
        per_principal["D_domain"][principal] += 1

    out = RESULTS / "benchmark_pairs.jsonl"
    out.write_text("\n".join(json.dumps(i) for i in instances) + "\n")

    summary = {
        "design": "EXPERIMENT_DESIGN.md / ROBUSTNESS_DESIGN.md",
        "pairs": dict(support),
        "instances": len(instances),
        "min_support_for_stats": MIN_SUPPORT_FOR_STATS,
        "underpowered_configs": sorted(
            c for c, n in support.items() if n < MIN_SUPPORT_FOR_STATS
        ),
        "principals_per_config": {c: dict(v) for c, v in per_principal.items()},
        "strata": dict(Counter(i["stratum"] for i in instances)),
    }
    (RESULTS / "benchmark_pairs_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"wrote {out}  ({len(instances)} instances, {sum(support.values())} pairs)")
    print(f"\n  {'config':14s} {'pairs':>6s}  principals")
    for config, n in support.most_common():
        flag = "  [UNDERPOWERED - descriptive only]" if n < MIN_SUPPORT_FOR_STATS else ""
        print(f"  {config:14s} {n:>6d}  {len(per_principal[config])}{flag}")
    print(f"\n  strata: {summary['strata']}")
    print("  all build-time validity gates passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
