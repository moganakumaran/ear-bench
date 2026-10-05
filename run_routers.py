#!/usr/bin/env python3
"""Run all routers over the entitlement benchmark and score them.

Split
-----
20% of TASKS (stratified by data domain, seed 0) are held out as a CALIBRATION
split used only to fit R2's similarity thresholds and R5's domain-tool rates.
Nothing is reported on it. All results come from the remaining 80%.
Splitting by task, not by instance, keeps a task's three conditions together so
no information leaks across the split.

Call budget
-----------
R1/R2/R5 need no model. R3 is permission-blind so it runs once per task.
R4 sees entitlements and must run per instance. R6 makes one
principal-independent extraction per task and reuses it for every principal;
R7 reuses R6's extraction entirely, costing nothing extra.

Run:
  ./.venv/bin/python run_routers.py                 # main, Haiku 4.5
  ./.venv/bin/python run_routers.py --limit 30      # smoke test
  ./.venv/bin/python run_routers.py --model sonnet --seeds 1 --transfer
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import yaml

from build_benchmark import POLICY, build_db_index, entitled_domains
from common import RESULTS
from llm import MAIN_MODEL, TRANSFER_MODEL, LLM
from metrics import RouteDecision, aggregate, score_instance
from routers import (
    GOVERNED,
    SQL_CORE,
    EmbeddingRouter,
    SemanticGraphRouter,
    r1_keyword,
    r3_llm,
    r4_llm,
    r6_decide,
    r6_extract,
    r7_decide,
)

CALIB_FRACTION = 0.20
SPLIT_SEED = 0
# R7's confidence threshold is FIT ON CALIBRATION, never on the evaluation
# split. Candidates span the range models actually emit.
R7_CANDIDATES = (0.0, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99)


def make_split(instances: list[dict]) -> tuple[set[str], set[str]]:
    """Stratified task-level calibration/eval split."""
    by_domain = defaultdict(list)
    for task_id, domain in {i["task_id"]: i["data_domain"] for i in instances}.items():
        by_domain[domain].append(task_id)
    calib: set[str] = set()
    rng = random.Random(SPLIT_SEED)
    for domain, tasks in sorted(by_domain.items()):
        tasks = sorted(tasks)
        rng.shuffle(tasks)
        k = max(1, round(len(tasks) * CALIB_FRACTION))
        calib.update(tasks[:k])
    everything = {i["task_id"] for i in instances}
    return calib, everything - calib


def fit_graph_rates(instances: list[dict], calib: set[str]) -> dict[str, dict[str, float]]:
    """R5 edge weights: P(tool required | domain), on calibration tasks only."""
    seen: dict[str, dict] = {}
    for i in instances:
        if i["task_id"] in calib and i["task_id"] not in seen:
            seen[i["task_id"]] = i
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    totals: dict[str, int] = defaultdict(int)
    for inst in seen.values():
        d = inst["data_domain"]
        totals[d] += 1
        for tool in inst["required_governed"]:
            counts[d][tool] += 1
    return {
        d: {t: counts[d][t] / totals[d] for t in GOVERNED} for d in totals
    }


def fit_embedding_thresholds(router: EmbeddingRouter, calib_tasks: list[dict]) -> dict:
    """Pick each governed tool's threshold to maximise F1 on calibration."""
    import numpy as np

    scores = router.scores([t["query"] for t in calib_tasks])
    out = {}
    for idx, tool in enumerate(GOVERNED):
        truth = np.array([tool in t["required_governed"] for t in calib_tasks])
        col = scores[:, idx]
        best, best_f1 = 0.25, -1.0
        for cand in np.unique(np.round(col, 3)):
            pred = col >= cand
            tp = int((pred & truth).sum())
            if tp == 0:
                f1 = 0.0
            else:
                prec = tp / max(int(pred.sum()), 1)
                rec = tp / max(int(truth.sum()), 1)
                f1 = 2 * prec * rec / (prec + rec)
            if f1 > best_f1:
                best, best_f1 = float(cand), f1
        out[tool] = {"threshold": round(best, 4), "calib_f1": round(best_f1, 4)}
    return out


def fit_r7_threshold(llm, calib_tasks, calib_instances, seed: int, workers: int = 8) -> dict:
    """Choose R7's confidence threshold on the calibration split.

    Objective is the unweighted mean of safe_rate and useful_rate: a gate that
    abstains too eagerly wins safety but loses usefulness, and a gate that never
    fires is just R6. Stating the objective up front stops the threshold from
    being chosen after seeing evaluation results.
    """
    def _one(t):
        data, _ = r6_extract(llm, t["query"], t["gold_db"], seed)
        return t["task_id"], data

    with ThreadPoolExecutor(max_workers=workers) as ex:
        need = dict(ex.map(_one, calib_tasks))
    confs = [
        (need[t["task_id"]] or {}).get("confidence")
        for t in calib_tasks
        if need.get(t["task_id"])
    ]
    numeric = [float(c) for c in confs if isinstance(c, (int, float))]

    trials = []
    for cand in R7_CANDIDATES:
        scored = [
            score_instance(i, r7_decide(need.get(i["task_id"]), i, cand))
            for i in calib_instances
        ]
        agg = aggregate(scored)
        safe = agg["safe_rate"] or 0.0
        useful = agg["useful_rate"] or 0.0
        trials.append(
            {
                "threshold": cand,
                "safe_rate": safe,
                "useful_rate": useful,
                "objective": round((safe + useful) / 2, 4),
            }
        )
    best = max(trials, key=lambda t: t["objective"])
    return {
        "chosen": best["threshold"],
        "objective": "mean(safe_rate, useful_rate)",
        "trials": trials,
        "calibration_confidence": {
            "n": len(numeric),
            "min": round(min(numeric), 3) if numeric else None,
            "max": round(max(numeric), 3) if numeric else None,
            "mean": round(sum(numeric) / len(numeric), 3) if numeric else None,
            "below_0.7": sum(1 for c in numeric if c < 0.7),
            "non_numeric": len(confs) - len(numeric),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="haiku", choices=["haiku", "sonnet"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N eval tasks by id")
    ap.add_argument(
        "--sample", type=int, default=None,
        help="evaluate a STRATIFIED random subsample of N eval tasks (seed 0). "
             "Used for the decode-variance arm, where a full sweep buys little.",
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--transfer", action="store_true", help="tag output as the transfer run")
    ap.add_argument("--benchmark", default="benchmark.jsonl")
    ap.add_argument(
        "--temperature", type=float, default=0.0,
        help="0.0 = deterministic primary run. >0 with --seeds>1 is the decode-variance arm.",
    )
    args = ap.parse_args()

    model = MAIN_MODEL if args.model == "haiku" else TRANSFER_MODEL

    instances = [json.loads(line) for line in (RESULTS / args.benchmark).open()]
    calib, eval_tasks = make_split(instances)
    if args.limit:
        eval_tasks = set(sorted(eval_tasks)[: args.limit])
    if args.sample:
        domain_of = {i["task_id"]: i["data_domain"] for i in instances}
        buckets: dict[str, list[str]] = defaultdict(list)
        for t in sorted(eval_tasks):
            buckets[domain_of[t]].append(t)
        rng = random.Random(SPLIT_SEED)
        picked: set[str] = set()
        share = args.sample / len(eval_tasks)
        for domain, tasks in sorted(buckets.items()):
            tasks = sorted(tasks)
            rng.shuffle(tasks)
            picked.update(tasks[: max(1, round(len(tasks) * share))])
        eval_tasks = picked
        tag_sample = f"_s{len(picked)}"
    else:
        tag_sample = ""

    tag = f"{args.model}{'_transfer' if args.transfer else ''}"
    if args.temperature > 0:
        tag += f"_t{args.temperature:g}"
    tag += tag_sample
    if args.benchmark != "benchmark.jsonl":
        tag += "_" + args.benchmark.replace("benchmark_", "").replace(".jsonl", "")

    eval_inst = [i for i in instances if i["task_id"] in eval_tasks]
    # One representative row per task for the permission-blind routers.
    tasks: dict[str, dict] = {}
    for i in eval_inst:
        tasks.setdefault(i["task_id"], i)
    task_list = sorted(tasks.values(), key=lambda t: t["task_id"])
    calib_tasks = sorted(
        {i["task_id"]: i for i in instances if i["task_id"] in calib}.values(),
        key=lambda t: t["task_id"],
    )

    print(f"benchmark      : {args.benchmark}")
    print(f"model          : {model}")
    print(f"calibration    : {len(calib_tasks)} tasks (not reported)")
    print(f"evaluation     : {len(task_list)} tasks / {len(eval_inst)} instances")
    print(f"seeds          : {args.seeds}\n")

    # --- fit the two calibrated routers -----------------------------------
    print("fitting R5 domain-tool rates and R2 thresholds on calibration ...")
    graph_rates = fit_graph_rates(instances, calib)
    r5 = SemanticGraphRouter(graph_rates)
    r2 = EmbeddingRouter()
    thresholds = fit_embedding_thresholds(r2, calib_tasks)
    r2.thresholds = {t: v["threshold"] for t, v in thresholds.items()}
    print("  R2 thresholds:", {t: v["threshold"] for t, v in thresholds.items()})

    # Each principal's full database scope, straight from the declared policy.
    policy = yaml.safe_load(POLICY.read_text())
    db_index = build_db_index(policy)
    entitled_dbs = {
        name: sorted(
            db for db, domain in db_index.items()
            if domain in entitled_domains(policy, name)
        )
        for name in policy["principals"]
    }

    r2_scores = r2.scores([t["query"] for t in task_list])
    r2_by_task = {t["task_id"]: r2.decide(r2_scores[n]) for n, t in enumerate(task_list)}
    r1_by_task = {t["task_id"]: r1_keyword(t["query"]) for t in task_list}
    r5_by_task = {t["task_id"]: r5.decide(t["data_domain"]) for t in task_list}

    llm = LLM(model=model, temperature=args.temperature, max_tokens=400)
    parse_errors: dict[str, int] = defaultdict(int)

    calib_inst = [i for i in instances if i["task_id"] in calib]
    print("calibrating R7 confidence gate on the calibration split ...")
    r7_cal = fit_r7_threshold(llm, calib_tasks, calib_inst, seed=0, workers=args.workers)
    r7_threshold = r7_cal["chosen"]
    print(f"  confidence on calibration: {r7_cal['calibration_confidence']}")
    print(f"  chosen threshold: {r7_threshold}")
    decisions: list[dict] = []
    started = time.time()

    for seed in range(args.seeds):
        print(f"\n--- seed {seed} ---")

        # R3: permission-blind, once per task.
        def do_r3(t):
            d, err = r3_llm(llm, t["query"], t["gold_db"], seed)
            return t["task_id"], d, err

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            r3_by_task = {}
            for tid, d, err in ex.map(do_r3, task_list):
                r3_by_task[tid] = d
                parse_errors["R3"] += err
        print(f"  R3 done ({len(r3_by_task)} tasks)")

        # R6: ONE principal-independent extraction per task.
        def do_r6(t):
            need, err = r6_extract(llm, t["query"], t["gold_db"], seed)
            return t["task_id"], need, err

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            need_by_task = {}
            for tid, need, err in ex.map(do_r6, task_list):
                need_by_task[tid] = need
                parse_errors["R6"] += err
        print(f"  R6 extraction done ({len(need_by_task)} tasks)")

        # R4: entitlement-aware prompt, per instance. The principal's FULL
        # database scope goes in the prompt — checking membership in a list of
        # up to 73 names is precisely what this baseline is being tested on.
        def do_r4(inst):
            # Under --opaque-principals the instance carries an opaque label;
            # the SCOPE must still be looked up by the declared principal, while
            # the name shown to the model stays opaque. That separation is the
            # whole point of the arm.
            # Paired instances (build_pairs.py) carry their OWN entitlement,
            # which by construction differs from the policy's view of that
            # principal. Prefer the instance's list when present; fall back to
            # the policy lookup for the main benchmark arms.
            if inst.get("entitled_databases") is not None:
                dbs = inst["entitled_databases"]
            else:
                scope_key = inst.get("principal_declared", inst["principal"])
                dbs = entitled_dbs[scope_key]
            d, err = r4_llm(llm, inst, dbs, seed)
            return inst["instance_id"], d, err

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            r4_by_inst = {}
            for iid, d, err in ex.map(do_r4, eval_inst):
                r4_by_inst[iid] = d
                parse_errors["R4"] += err
        print(f"  R4 done ({len(r4_by_inst)} instances)")

        for inst in eval_inst:
            tid, iid = inst["task_id"], inst["instance_id"]
            blind = {
                "R1_keyword": r1_by_task[tid],
                "R2_embedding": r2_by_task[tid],
                "R3_llm_blind": r3_by_task[tid],
                "R5_semantic_graph": r5_by_task[tid],
            }
            for name, dec in blind.items():
                # Permission-blind routers name the database they would query.
                decisions.append(
                    _row(name, seed, inst, RouteDecision(
                        abstained=dec.abstained, tools=dec.tools,
                        database=inst["gold_db"], rationale=dec.rationale))
                )
            decisions.append(_row("R4_llm_entitled", seed, inst, r4_by_inst[iid]))
            need = need_by_task[tid]
            decisions.append(_row("R6_entitlement_aware", seed, inst, r6_decide(need, inst)))
            decisions.append(
                _row("R7_confidence_gated", seed, inst,
                     r7_decide(need, inst, r7_threshold))
            )

        print(f"  usage: {llm.usage.summary(model)}")

    out_dir = RESULTS
    (out_dir / f"decisions_{tag}.jsonl").write_text(
        "\n".join(json.dumps(d) for d in decisions) + "\n"
    )

    summary = _summarise(decisions)
    meta = {
        "model": model,
        "seeds": args.seeds,
        "temperature": args.temperature,
        "benchmark": args.benchmark,
        "eval_tasks": len(task_list),
        "eval_instances": len(eval_inst),
        "calibration_tasks": len(calib_tasks),
        "calibration_fraction": CALIB_FRACTION,
        "split_seed": SPLIT_SEED,
        "r7_calibration": r7_cal,
        "entitled_db_counts": {k: len(v) for k, v in entitled_dbs.items()},
        "r2_thresholds": thresholds,
        "r5_domain_tool_rates": graph_rates,
        "parse_errors": dict(parse_errors),
        "usage": llm.usage.summary(model),
        "wall_seconds": round(time.time() - started, 1),
        "results": summary,
    }
    (out_dir / f"router_results_{tag}.json").write_text(json.dumps(meta, indent=2))

    print(f"\nwrote {out_dir}/decisions_{tag}.jsonl and router_results_{tag}.json\n")
    _print_table(summary)
    return 0


def _row(router: str, seed: int, inst: dict, dec: RouteDecision) -> dict:
    scored = score_instance(inst, dec)
    scored.update(
        {
            "router": router,
            "seed": seed,
            "tools": sorted(dec.tools),
            "decision_abstained": dec.abstained,
            "rationale": dec.rationale,
        }
    )
    return scored


def _summarise(decisions: list[dict]) -> dict:
    out: dict = {}
    routers = sorted({d["router"] for d in decisions})
    for r in routers:
        rows = [d for d in decisions if d["router"] == r]
        entry = {"overall": aggregate(rows)}
        per_seed = []
        for s in sorted({d["seed"] for d in rows}):
            per_seed.append(aggregate([d for d in rows if d["seed"] == s]))
        entry["per_seed"] = per_seed
        for cond in ("A_unrestricted", "B_natural", "C_foreign"):
            entry[cond] = aggregate([d for d in rows if d["condition"] == cond])
        for st in ("FEASIBLE", "INFEASIBLE_SCOPE", "INFEASIBLE_TOOL"):
            entry[st] = aggregate([d for d in rows if d["stratum"] == st])
        out[r] = entry
    return out


def _print_table(summary: dict) -> None:
    hdr = f"{'router':24s} {'over_reach':>11s} {'safe':>7s} {'useful':>7s} {'tool_f1':>8s} {'abs_prec':>9s} {'abs_rec':>8s}"
    print(hdr)
    print("-" * len(hdr))
    for name, entry in summary.items():
        o = entry["overall"]
        def f(x):
            return f"{x:.4f}" if isinstance(x, (int, float)) else "   n/a"
        print(
            f"{name:24s} {f(o['over_reach_rate']):>11s} {f(o['safe_rate']):>7s} "
            f"{f(o['useful_rate']):>7s} {f(o['tool_set_f1']):>8s} "
            f"{f(o['abstention_precision']):>9s} {f(o['abstention_recall']):>8s}"
        )


if __name__ == "__main__":
    sys.exit(main())
