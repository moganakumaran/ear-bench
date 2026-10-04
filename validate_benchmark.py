#!/usr/bin/env python3
"""Independent validation of the entitlement-extended benchmark.

Deliberately re-derives its expectations from the policy and the raw parquet
rather than trusting build_benchmark.py's own summary, so a bug in the builder
cannot validate itself. Exit 0 = benchmark is sound.

Run:  ./.venv/bin/python validate_benchmark.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter

import yaml

from build_benchmark import POLICY, build_db_index, build_owner_index, entitled_domains
from common import GOVERNED_TOOLS, RESULTS, load_labelled

# Expectations fixed from Day-1 characterisation, before any router was run.
EXPECT_TASKS = 834
EXPECT_TOTAL = 2007
EXPECT_DATABASES = 73
EXPECT_CONDITIONS = 3

FAILURES: list[str] = []
CHECKS = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if ok:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}{(' — ' + detail) if detail else ''}")
        FAILURES.append(name)


def main() -> int:
    policy = yaml.safe_load(POLICY.read_text())
    df = load_labelled()
    db_index = build_db_index(policy)
    owners = build_owner_index(policy)
    inst = [json.loads(line) for line in (RESULTS / "benchmark.jsonl").open()]

    print("Entitlement benchmark validation\n")

    # --- shape -------------------------------------------------------------
    check("labelled task count is 834", len(df) == EXPECT_TASKS, str(len(df)))
    check(
        "instance count is tasks x conditions",
        len(inst) == EXPECT_TASKS * EXPECT_CONDITIONS,
        str(len(inst)),
    )
    check(
        "every task appears under every condition",
        all(c == EXPECT_CONDITIONS for c in Counter(i["task_id"] for i in inst).values()),
    )
    check("instance ids are unique", len({i["instance_id"] for i in inst}) == len(inst))

    # --- policy integrity --------------------------------------------------
    check("policy declares 73 databases", len(db_index) == EXPECT_DATABASES, str(len(db_index)))
    check(
        "policy covers exactly the observed databases",
        set(db_index) == set(df["gold_db"]),
    )
    check(
        "every domain has exactly one owning principal",
        set(owners) == set(policy["data_domains"]),
    )
    check(
        "governed tools in policy match common.GOVERNED_TOOLS",
        set(policy["governed_tools"]) == set(GOVERNED_TOOLS),
    )

    # --- stratum logic re-derived from scratch -----------------------------
    mislabelled = []
    for i in inst:
        held = set(policy["principals"][i["principal"]]["governed_tools"])
        dom_ok = i["data_domain"] in entitled_domains(policy, i["principal"])
        missing = [t for t in i["required_governed"] if t not in held]
        if not dom_ok:
            expect = "INFEASIBLE_SCOPE"
        elif missing:
            expect = "INFEASIBLE_TOOL"
        else:
            expect = "FEASIBLE"
        if expect != i["stratum"]:
            mislabelled.append((i["instance_id"], i["stratum"], expect))
    check(
        "every stratum re-derives from the policy",
        not mislabelled,
        f"{len(mislabelled)} mismatch(es), e.g. {mislabelled[:1]}",
    )

    # --- target consistency ------------------------------------------------
    check(
        "infeasible instances target ABSTAIN",
        all(i["entitled_gold_tools"] == "ABSTAIN" for i in inst if i["stratum"] != "FEASIBLE"),
    )
    check(
        "feasible instances target the full gold tool set",
        all(
            i["entitled_gold_tools"] == i["gold_tools"]
            for i in inst
            if i["stratum"] == "FEASIBLE"
        ),
    )
    check(
        "entitled_gold_db is set iff the database is entitled",
        all((i["entitled_gold_db"] is not None) == i["database_entitled"] for i in inst),
    )
    check(
        "no feasible instance has an unentitled database",
        not [i for i in inst if i["stratum"] == "FEASIBLE" and not i["database_entitled"]],
    )

    # --- the control arm ---------------------------------------------------
    arm_a = [i for i in inst if i["condition"] == "A_unrestricted"]
    check("condition A covers every task", len(arm_a) == EXPECT_TASKS)
    check("condition A is entirely feasible", all(i["stratum"] == "FEASIBLE" for i in arm_a))
    gold_by_task = dict(zip(df["task_id"], df["gold_tool_set"]))
    check(
        "condition A reproduces the base gold route exactly",
        all(tuple(i["entitled_gold_tools"]) == gold_by_task[i["task_id"]] for i in arm_a),
    )
    check(
        "condition A uses only platform_admin",
        {i["principal"] for i in arm_a} == {"platform_admin"},
    )

    # --- the other arms ----------------------------------------------------
    arm_b = [i for i in inst if i["condition"] == "B_natural"]
    check(
        "condition B always assigns the domain owner",
        all(i["principal"] == owners[i["data_domain"]] for i in arm_b),
    )
    arm_c = [i for i in inst if i["condition"] == "C_foreign"]
    check(
        "condition C is entirely out of data scope",
        all(i["stratum"] == "INFEASIBLE_SCOPE" for i in arm_c),
    )
    check(
        "condition C never assigns platform_admin",
        "platform_admin" not in {i["principal"] for i in arm_c},
    )
    spread = Counter(i["principal"] for i in arm_c)
    check(
        "condition C spreads across all six restricted principals",
        len(spread) == 6,
        str(dict(spread)),
    )

    # --- determinism -------------------------------------------------------
    from build_benchmark import pick_foreign

    check(
        "foreign-principal choice is reproducible",
        all(
            pick_foreign(i["task_id"], i["data_domain"], policy) == i["principal"]
            for i in arm_c
        ),
    )

    # --- measurability -----------------------------------------------------
    strata = Counter(i["stratum"] for i in inst)
    check(
        "every stratum is large enough to estimate a rate (n >= 100)",
        all(v >= 100 for v in strata.values()),
        str(dict(strata)),
    )

    print(f"\n{CHECKS - len(FAILURES)}/{CHECKS} checks passed")
    if FAILURES:
        print("FAILED: " + ", ".join(FAILURES))
        return 1
    print("strata:", dict(strata))
    return 0


if __name__ == "__main__":
    sys.exit(main())
