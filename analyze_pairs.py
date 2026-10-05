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

    (RESULTS / "pairs_analysis.json").write_text(json.dumps(out, indent=2))

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
