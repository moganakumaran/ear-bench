#!/usr/bin/env python3
"""Build the entitlement-extended routing benchmark.

Takes the 834 labelled FDABench tasks and the declared policy in
entitlements.yaml, and emits one instance per (task, condition) with its
entitled gold route and feasibility stratum.

The extension is ADDITIVE and REVERSIBLE: under condition A_unrestricted the
entitled gold route is identical to the base gold route, so that arm reproduces
base-benchmark routing behaviour and serves as the harness correctness check.

Strata
------
FEASIBLE          principal may query the task's database and holds every
                  governed tool the gold route requires. Correct route = gold.
INFEASIBLE_SCOPE  principal may not query the task's database. Because
                  execute_sql and get_schema_info appear in 100% of gold routes,
                  no entitled route exists. Correct action = abstain.
INFEASIBLE_TOOL   principal may query the database but lacks >=1 required
                  governed tool. The gold route cannot be completed.
                  Correct action = abstain.

Run:  ./.venv/bin/python build_benchmark.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import yaml

from common import GOVERNED_TOOLS, RESULTS, canon_db, config_totals, load_labelled

HERE = Path(__file__).resolve().parent
POLICY = HERE / "entitlements.yaml"

ABSTAIN = "ABSTAIN"


class PolicyError(RuntimeError):
    """The declared policy does not cover the data. Always fatal."""


def load_policy() -> dict:
    return yaml.safe_load(POLICY.read_text())


def build_db_index(policy: dict) -> dict[str, str]:
    """database -> domain, asserting each database is declared exactly once."""
    index: dict[str, str] = {}
    dupes: list[str] = []
    for domain, spec in policy["data_domains"].items():
        for raw in spec["databases"]:
            db = canon_db(raw)
            if db in index:
                dupes.append(f"{raw!r} in both {index[db]!r} and {domain!r}")
            index[db] = domain
    if dupes:
        raise PolicyError("databases declared in more than one domain: " + "; ".join(dupes))
    return index


def shuffle_db_index(index: dict[str, str], seed: int) -> dict[str, str]:
    """Reassign databases to domains at random, preserving each domain's size.

    Experimental control. In the declared policy a database's domain is
    semantically predictable from the query — a COVID question obviously belongs
    to health_bio — so a router could appear entitlement-aware while really just
    guessing the domain from query content. Shuffling severs that correlation,
    isolating whether a router actually consults the entitlement list.

    Preserves the NUMBER OF DATABASES per domain, not the number of tasks or the
    stratum sizes: task counts and governed-tool requirements vary by database,
    so e.g. INFEASIBLE_TOOL moves from 197 (declared) to 121 (seed 0). Compare
    arms by rate, never by raw count.
    """
    dbs = sorted(index)
    domains = [index[db] for db in dbs]  # multiset of domain labels, sizes preserved
    rng = random.Random(seed)
    rng.shuffle(domains)
    return dict(zip(dbs, domains))


def check_coverage(index: dict[str, str], observed: set[str]) -> None:
    missing = sorted(observed - set(index))
    if missing:
        raise PolicyError(
            f"{len(missing)} database(s) appear in gold routes but are not "
            f"declared in entitlements.yaml: {missing}"
        )
    unused = sorted(set(index) - observed)
    if unused:
        raise PolicyError(
            f"{len(unused)} database(s) are declared but never appear in the "
            f"data — likely a typo: {unused}"
        )


def build_owner_index(policy: dict) -> dict[str, str]:
    """domain -> owning principal, asserting exactly one owner per domain."""
    owners: dict[str, str] = {}
    for name, spec in policy["principals"].items():
        for domain in spec.get("owns") or []:
            if domain in owners:
                raise PolicyError(
                    f"domain {domain!r} is owned by both {owners[domain]!r} and {name!r}"
                )
            owners[domain] = name
    undeclared = sorted(set(policy["data_domains"]) - set(owners))
    if undeclared:
        raise PolicyError(f"domains with no owning principal: {undeclared}")
    return owners


def entitled_domains(policy: dict, principal: str) -> set[str]:
    spec = policy["principals"][principal]["data_domains"]
    if spec == "ALL":
        return set(policy["data_domains"])
    return set(spec)


def pick_foreign(task_id: str, domain: str, policy: dict) -> str:
    """Deterministically choose a principal NOT entitled to `domain`.

    Stable across runs and machines: selection is by SHA-256 of the task id, not
    by Python's randomised hash.
    """
    candidates = sorted(
        name
        for name in policy["principals"]
        if name != "platform_admin" and domain not in entitled_domains(policy, name)
    )
    if not candidates:
        raise PolicyError(f"no principal is foreign to domain {domain!r}")
    digest = hashlib.sha256(task_id.encode()).digest()
    return candidates[int.from_bytes(digest[:8], "big") % len(candidates)]


def classify(
    policy: dict, principal: str, domain: str, required_governed: tuple[str, ...]
) -> tuple[str, list[str]]:
    """Return (stratum, missing_governed_tools)."""
    if domain not in entitled_domains(policy, principal):
        return "INFEASIBLE_SCOPE", []
    held = set(policy["principals"][principal]["governed_tools"])
    missing = [t for t in required_governed if t not in held]
    if missing:
        return "INFEASIBLE_TOOL", missing
    return "FEASIBLE", []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--opaque-principals",
        action="store_true",
        help=(
            "Rename principals to principal_1..principal_N under the DECLARED "
            "assignment. Severs the principal-name semantic cue without inducing "
            "policy-semantics conflict, and unlike --policy-seed it is a genuine "
            "one-factor manipulation: strata sizes, the calibration split and "
            "every principal's scope are preserved exactly. See PREREGISTRATION.md."
        ),
    )
    ap.add_argument(
        "--policy-seed",
        type=int,
        default=None,
        help=(
            "Shuffle the database->domain assignment with this seed, preserving "
            "domain sizes. Writes benchmark_shuffled_<seed>.jsonl. Used as the "
            "control arm that breaks the semantic predictability of domains."
        ),
    )
    args = ap.parse_args()

    policy = load_policy()
    df = load_labelled()

    db_index = build_db_index(policy)
    check_coverage(db_index, set(df["gold_db"]))
    owners = build_owner_index(policy)

    # Opaque principal names. The map is sorted so it is stable across runs;
    # platform_admin keeps a distinct index like any other principal.
    opaque: dict[str, str] = {}
    if args.opaque_principals:
        opaque = {
            name: f"principal_{i}"
            for i, name in enumerate(sorted(policy["principals"]), start=1)
        }

    suffix = ""
    if args.opaque_principals:
        suffix = "_opaque"
    if args.policy_seed is not None:
        db_index = shuffle_db_index(db_index, args.policy_seed)
        suffix = f"_shuffled_{args.policy_seed}"

    instances = []
    for _, row in df.iterrows():
        domain = db_index[row["gold_db"]]
        assignment = {
            "A_unrestricted": "platform_admin",
            "B_natural": owners[domain],
            "C_foreign": pick_foreign(row["task_id"], domain, policy),
        }
        for condition, principal in assignment.items():
            stratum, missing = classify(
                policy, principal, domain, row["required_governed"]
            )
            allowed_db = stratum != "INFEASIBLE_SCOPE"
            instances.append(
                {
                    "instance_id": f"{row['task_id']}::{condition}",
                    "task_id": row["task_id"],
                    "config": row["config"],
                    "level": row["level"],
                    "question_type": row["question_type"],
                    "database_type": row["database_type"],
                    "query": row["query"],
                    "gold_db": row["gold_db"],
                    "data_domain": domain,
                    "gold_tools": list(row["gold_tool_set"]),
                    "required_governed": list(row["required_governed"]),
                    "condition": condition,
                    "principal": opaque.get(principal, principal),
                    "principal_declared": principal,
                    "entitled_domains": sorted(entitled_domains(policy, principal)),
                    "entitled_databases_count": sum(
                        1 for d in db_index.values()
                        if d in entitled_domains(policy, principal)
                    ),
                    "entitled_governed_tools": sorted(
                        policy["principals"][principal]["governed_tools"]
                    ),
                    "stratum": stratum,
                    "missing_governed_tools": missing,
                    "database_entitled": allowed_db,
                    # The target the router is scored against.
                    "entitled_gold_tools": (
                        list(row["gold_tool_set"]) if stratum == "FEASIBLE" else ABSTAIN
                    ),
                    "entitled_gold_db": row["gold_db"] if allowed_db else None,
                    "db_column_disagrees": bool(row["db_column_disagrees"]),
                }
            )

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"benchmark{suffix}.jsonl"
    with out.open("w") as fh:
        for inst in instances:
            fh.write(json.dumps(inst) + "\n")

    # --- Correctness check: condition A must be a faithful pass-through -----
    arm_a = [i for i in instances if i["condition"] == "A_unrestricted"]
    bad = [
        i for i in arm_a
        if i["stratum"] != "FEASIBLE" or i["entitled_gold_tools"] != i["gold_tools"]
    ]
    if bad:
        raise PolicyError(
            f"condition A is not a faithful pass-through: {len(bad)} instance(s) "
            f"differ from the base gold route, e.g. {bad[0]['instance_id']}"
        )

    strata = Counter((i["condition"], i["stratum"]) for i in instances)
    summary = {
        "base_benchmark": "FDABench (arXiv:2509.02473, KDD'26)",
        "policy_version": policy["version"],
        "policy_seed": args.policy_seed,
        "tasks_labelled": len(df),
        "tasks_total": sum(v["total"] for v in config_totals().values()),
        "instances": len(instances),
        "conditions": sorted({i["condition"] for i in instances}),
        "strata_by_condition": {
            cond: {
                s: strata[(cond, s)]
                for s in ("FEASIBLE", "INFEASIBLE_SCOPE", "INFEASIBLE_TOOL")
                if strata[(cond, s)]
            }
            for cond in sorted({i["condition"] for i in instances})
        },
        "strata_overall": dict(Counter(i["stratum"] for i in instances)),
        "principal_usage": dict(Counter(i["principal"] for i in instances)),
        "domain_task_counts": dict(Counter(db_index[d] for d in df["gold_db"])),
        "databases_declared": len(db_index),
        "databases_observed": int(df["gold_db"].nunique()),
        "tasks_with_db_column_disagreement": int(df["db_column_disagrees"].sum()),
        "governed_tools": list(GOVERNED_TOOLS),
        "condition_A_passthrough_verified": True,
    }
    (RESULTS / f"benchmark_summary{suffix}.json").write_text(json.dumps(summary, indent=2))

    print(f"wrote {out}  ({len(instances)} instances from {len(df)} tasks)")
    print(f"wrote {RESULTS}/benchmark_summary{suffix}.json")
    print("\nstrata by condition:")
    for cond, counts in summary["strata_by_condition"].items():
        total = sum(counts.values())
        detail = "  ".join(f"{k}={v}" for k, v in counts.items())
        print(f"  {cond:16s} n={total:4d}   {detail}")
    print("\noverall:", summary["strata_overall"])
    print("condition A pass-through verified: entitled gold == base gold on all 834")


if __name__ == "__main__":
    main()
