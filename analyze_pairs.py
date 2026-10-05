#!/usr/bin/env python3
"""Analysis for the paired permit/revoke control. Zero API calls.

Tests are exactly those fixed in EXPERIMENT_DESIGN.md section 7, written before
the experiment ran:

  1. Within-router, across arms: exact McNemar on invoked(g), PERMIT vs REVOKE,
     paired on (task, principal). This is the one-factor test -- the correct
     answer differs between arms, so over-reach cannot be compared across them,
     but "did the router invoke the tool" is defined identically in both.
  2. Across routers, within REVOKE: exact McNemar on over-reach, R4 vs R6.
  3. Holm adjustment across the family reported here.

Configurations below MIN_SUPPORT_FOR_STATS are reported descriptively with no
interval and no test (ROBUSTNESS_DESIGN.md section 3).

Run:  ./.venv/bin/python analyze_pairs.py
"""
from __future__ import annotations

import json
from collections import defaultdict

from common import RESULTS
from routers import SQL_CORE
from stats import holm, mcnemar, wilson

MIN_SUPPORT_FOR_STATS = 100
R4, R6 = "R4_llm_entitled", "R6_entitlement_aware"
# Which tool each tool-axis config revokes, for the invocation test.
REVOKED = {
    "A_web": ("web_search",),
    "B_vector": ("vector_search",),
    "C_file": ("file_system",),
    "D_web_vector": ("web_search", "vector_search"),
}


def _write_principal_table(per_principal: dict) -> None:
    """Artifact-only Markdown table backing the 'four principals' statement."""
    lines = ["# Principal-level breakdown of the paired permit/revoke control",
             "",
             "Artifact table referenced by the manuscript. Evaluation pairs only;",
             "calibration tasks are held out. Percentages with 95% Wilson intervals.",
             ""]
    for cfg, cells in per_principal.items():
        rows = [(p, c) for p, c in cells.items() if c["n_revoke"]]
        if not rows:
            continue
        lines += [f"## {cfg}", "",
                  "| principal | n | R4 over-reach | R4 false abst. | R6 over-reach | R6 false abst. |",
                  "|---|---:|---|---|---|---|"]
        def f(d):
            return f"{100*d['point']:.1f} [{100*d['ci_low']:.1f}, {100*d['ci_high']:.1f}]"
        for p, c in sorted(rows, key=lambda kv: -kv[1]["n_revoke"]):
            lines.append(f"| {p} | {c['n_revoke']} | {f(c['r4_over_reach'])} | "
                         f"{f(c['r4_false_abstention'])} | {f(c['r6_over_reach'])} | "
                         f"{f(c['r6_false_abstention'])} |")
        lines.append("")
    (RESULTS / "PRINCIPAL_BREAKDOWN.md").write_text("\n".join(lines))


def main() -> int:
    bench = {i["instance_id"]: i
             for i in map(json.loads, (RESULTS / "benchmark_pairs.jsonl").open())}
    rows = [json.loads(l) for l in (RESULTS / "decisions_haiku_pairs.jsonl").open()]
    by = defaultdict(dict)
    for r in rows:
        by[r["router"]][r["instance_id"]] = r

    summary = json.loads((RESULTS / "benchmark_pairs_summary.json").read_text())
    configs = sorted(summary["pairs"])
    out: dict = {"min_support_for_stats": MIN_SUPPORT_FOR_STATS, "configs": {}}
    raw_p: dict[str, float] = {}

    for cfg in configs:
        n_pairs = summary["pairs"][cfg]
        entry = {"pairs": n_pairs, "powered": n_pairs >= MIN_SUPPORT_FOR_STATS,
                 "routers": {}}
        for router in (R4, R6):
            dec = by[router]
            permit = [i for i in bench.values()
                      if i["config"] == cfg and i["arm"] == "PERMIT" and i["instance_id"] in dec]
            revoke = [i for i in bench.values()
                      if i["config"] == cfg and i["arm"] == "REVOKE" and i["instance_id"] in dec]

            over = wilson(sum(dec[i["instance_id"]]["over_reach"] for i in revoke), len(revoke))
            fabs = wilson(sum(dec[i["instance_id"]]["abstained"] for i in permit), len(permit))
            r = {"n_revoke": len(revoke), "n_permit": len(permit),
                 "over_reach_revoke": over.as_dict(),
                 "false_abstention_permit": fabs.as_dict()}

            # Test 1: invocation of the revoked tool, paired across arms.
            if cfg in REVOKED:
                tools = set(REVOKED[cfg])
                def inv(insts):
                    return [{"instance_id": i["pair_id"],
                             "x": bool(tools & (set(dec[i["instance_id"]]["tools"]) - set(SQL_CORE)))}
                            for i in insts]
                res = mcnemar(inv(permit), inv(revoke), "x")
                r["invoked_permit"] = round(
                    sum(v["x"] for v in inv(permit)) / max(len(permit), 1), 4)
                r["invoked_revoke"] = round(
                    sum(v["x"] for v in inv(revoke)) / max(len(revoke), 1), 4)
                r["mcnemar_invocation"] = res
                if entry["powered"] and "p_value" in res:
                    raw_p[f"{cfg}|{router}|invocation"] = res["p_value"]
            entry["routers"][router] = r

        # Test 2: R4 vs R6 over-reach inside REVOKE.
        rev_ids = [i["instance_id"] for i in bench.values()
                   if i["config"] == cfg and i["arm"] == "REVOKE"]
        a = [by[R4][i] for i in rev_ids if i in by[R4]]
        b = [by[R6][i] for i in rev_ids if i in by[R6]]
        res = mcnemar(a, b, "over_reach")
        entry["mcnemar_R4_vs_R6_over_reach"] = res
        if entry["powered"] and "p_value" in res:
            raw_p[f"{cfg}|R4_vs_R6|over_reach"] = res["p_value"]
        out["configs"][cfg] = entry

    for label, p_adj in holm(raw_p).items():
        cfg, who, what = label.split("|")
        if who == "R4_vs_R6":
            out["configs"][cfg]["mcnemar_R4_vs_R6_over_reach"]["p_holm"] = p_adj
        else:
            out["configs"][cfg]["routers"][who]["mcnemar_invocation"]["p_holm"] = p_adj

    # --- drift guard -------------------------------------------------------
    # The manuscript quotes both the BUILT pair count and the per-configuration
    # n in the results table. Those differ because calibration tasks are held
    # out. Assert the reported totals equal the evaluated pairs so the two can
    # never silently diverge again.
    reported = sum(e["routers"][R4]["n_revoke"] for e in out["configs"].values())
    evaluated = sum(
        1 for i in bench.values()
        if i["arm"] == "REVOKE" and i["instance_id"] in by[R4]
    )
    built = sum(summary["pairs"].values())
    if reported != evaluated:
        raise AssertionError(
            f"reported n ({reported}) != evaluated pairs ({evaluated})"
        )
    out["pair_accounting"] = {
        "built": built,
        "evaluated": evaluated,
        "held_out_in_calibration": built - evaluated,
        "reported_sum": reported,
    }

    # --- principal-level breakdown (artifact only, not the manuscript) ------
    per_principal: dict[str, dict] = {}
    for cfg in configs:
        cells: dict[str, dict] = {}
        for iid, inst in bench.items():
            if inst["config"] != cfg:
                continue
            pr = inst["principal"]
            c = cells.setdefault(pr, {"n_revoke": 0, "r4_over": 0, "r6_over": 0,
                                      "n_permit": 0, "r4_fabs": 0, "r6_fabs": 0})
            if inst["arm"] == "REVOKE" and iid in by[R4]:
                c["n_revoke"] += 1
                c["r4_over"] += bool(by[R4][iid]["over_reach"])
                c["r6_over"] += bool(by[R6][iid]["over_reach"])
            elif inst["arm"] == "PERMIT" and iid in by[R4]:
                c["n_permit"] += 1
                c["r4_fabs"] += bool(by[R4][iid]["abstained"])
                c["r6_fabs"] += bool(by[R6][iid]["abstained"])
        for pr, c in cells.items():
            if not c["n_revoke"]:
                continue
            c["r4_over_reach"] = wilson(c["r4_over"], c["n_revoke"]).as_dict()
            c["r6_over_reach"] = wilson(c["r6_over"], c["n_revoke"]).as_dict()
            c["r4_false_abstention"] = wilson(c["r4_fabs"], max(c["n_permit"], 1)).as_dict()
            c["r6_false_abstention"] = wilson(c["r6_fabs"], max(c["n_permit"], 1)).as_dict()
        per_principal[cfg] = cells
    out["per_principal"] = per_principal

    (RESULTS / "pairs_analysis.json").write_text(json.dumps(out, indent=2))
    _write_principal_table(per_principal)

    # ---------------- report ----------------
    def pc(d):
        return f"{100*d['point']:5.1f} [{100*d['ci_low']:4.1f},{100*d['ci_high']:5.1f}]"

    print("PAIRED PERMIT/REVOKE CONTROL\n")
    hdr = (f"  {'config':14s} {'n':>5s} {'router':>4s} "
           f"{'over-reach (REVOKE)':>21s} {'false abst (PERMIT)':>21s}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for cfg in configs:
        e = out["configs"][cfg]
        mark = "" if e["powered"] else "  *underpowered"
        for router, tag in ((R4, "R4"), (R6, "R6")):
            r = e["routers"][router]
            print(f"  {cfg if tag=='R4' else '':14s} {r['n_revoke']:>5d} {tag:>4s} "
                  f"{pc(r['over_reach_revoke']):>21s} {pc(r['false_abstention_permit']):>21s}"
                  f"{mark if tag=='R4' else ''}")
    print("\n  invocation of the revoked tool, paired across arms (R4)")
    for cfg in configs:
        r = out["configs"][cfg]["routers"][R4]
        if "mcnemar_invocation" not in r:
            continue
        m = r["mcnemar_invocation"]
        p = m.get("p_holm", m.get("p_value"))
        print(f"    {cfg:14s} PERMIT {100*r['invoked_permit']:5.1f}%  ->  "
              f"REVOKE {100*r['invoked_revoke']:5.1f}%   "
              f"permit_only={m.get('a_only','-'):>4} revoke_only={m.get('b_only','-'):>4}  "
              f"p={p if isinstance(p,str) else f'{p:.2e}'}")
    print(f"\nwrote {RESULTS}/pairs_analysis.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
