#!/usr/bin/env python3
"""Paired tests ACROSS arms, and an interval on the usefulness difference.

The manuscript asserted three comparisons with no test behind them:
  declared -> shuffled policy      (the headline)
  Haiku -> Sonnet backbone         ("scale does not fix it")
  R6 vs R4 usefulness              ("no measurable cost")

The first two are tested here. The third is not a test but an interval: a
non-significant McNemar is absence of evidence, so we report a confidence
interval on the paired difference and say what it excludes.

Arms differ in which instances exist (shuffling moves the calibration split), so
every cross-arm test is restricted to instances present in BOTH arms and is
reported with that n.

Run:  ./.venv/bin/python cross_arm.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from scipy import stats as sps

from common import RESULTS
from stats import holm, mcnemar, wilson

R4 = "R4_llm_entitled"
R6 = "R6_entitlement_aware"


def load(tag: str) -> list[dict]:
    path = RESULTS / f"decisions_{tag}.jsonl"
    return [json.loads(l) for l in path.open()] if path.exists() else []


def arm(rows: list[dict], router: str, stratum: str | None = None) -> list[dict]:
    out = [r for r in rows if r["router"] == router]
    return [r for r in out if r["stratum"] == stratum] if stratum else out


def paired_diff_ci(a: list[dict], b: list[dict], key: str, conf: float = 0.95) -> dict:
    """CI on the mean paired difference (a - b). Reports what can be EXCLUDED."""
    am = {r["instance_id"]: r for r in a}
    bm = {r["instance_id"]: r for r in b}
    ids = [i for i in am if i in bm
           and am[i].get(key) is not None and bm[i].get(key) is not None]
    d = [bool(am[i][key]) - bool(bm[i][key]) for i in ids]
    n = len(d)
    if n < 2:
        return {"n": n}
    mean = sum(d) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1))
    z = sps.norm.ppf(1 - (1 - conf) / 2)
    half = z * sd / math.sqrt(n)
    return {
        "n": n,
        "mean_pp": round(100 * mean, 2),
        "ci_low_pp": round(100 * (mean - half), 2),
        "ci_high_pp": round(100 * (mean + half), 2),
    }


def main() -> int:
    out: dict = {}
    raw: dict[str, float] = {}

    declared = load("haiku")
    sonnet = load("sonnet_transfer")

    # --- 1. declared vs each shuffled seed, and vs opaque ------------------
    for tag, label in [("haiku_shuffled_0", "shuffled_seed0"),
                       ("haiku_shuffled_1", "shuffled_seed1"),
                       ("haiku_shuffled_2", "shuffled_seed2"),
                       ("haiku_opaque", "opaque_principals")]:
        rows = load(tag)
        if not rows:
            continue
        out[label] = {}
        for router in (R4, R6):
            entry = {}
            for stratum, key in (("INFEASIBLE_SCOPE", "over_reach"),
                                 ("FEASIBLE", "abstained")):
                a = arm(rows, router, stratum)
                b = arm(declared, router, stratum)
                res = mcnemar(a, b, key)
                wa, wb = (wilson(sum(bool(r[key]) for r in x), len(x)) for x in (a, b))
                entry[f"{stratum}:{key}"] = {
                    "declared": wa.as_dict() and wb.as_dict(),
                    "arm": wa.as_dict(),
                    "declared_rate": wb.as_dict(),
                    "mcnemar": res,
                }
                if "p_value" in res:
                    raw[f"{label}|{router}|{stratum}:{key}"] = res["p_value"]
            out[label][router] = entry

    # --- 2. backbone transfer ---------------------------------------------
    if sonnet:
        out["backbone_transfer"] = {}
        for router in (R4, R6, "R3_llm_blind"):
            res = mcnemar(arm(sonnet, router), arm(declared, router), "over_reach")
            out["backbone_transfer"][router] = {
                "haiku": wilson(sum(r["over_reach"] for r in arm(declared, router)),
                                len(arm(declared, router))).as_dict(),
                "sonnet": wilson(sum(r["over_reach"] for r in arm(sonnet, router)),
                                 len(arm(sonnet, router))).as_dict(),
                "mcnemar": res,
            }
            if "p_value" in res:
                raw[f"backbone|{router}|over_reach"] = res["p_value"]

    # --- 3. what the usefulness comparison can exclude --------------------
    out["usefulness_interval"] = {
        "R6_minus_R4": paired_diff_ci(arm(declared, R6), arm(declared, R4), "useful"),
    }

    for label, p_adj in holm(raw).items():
        parts = label.split("|")
        if parts[0] == "backbone":
            out["backbone_transfer"][parts[1]]["mcnemar"]["p_holm"] = p_adj
        else:
            out[parts[0]][parts[1]][parts[2]]["mcnemar"]["p_holm"] = p_adj

    (RESULTS / "cross_arm.json").write_text(json.dumps(out, indent=2))

    # --- report -----------------------------------------------------------
    print("cross-arm paired tests (Holm-adjusted across all comparisons here)\n")
    for label in [k for k in out if k.startswith(("shuffled", "opaque"))]:
        print(f"{label}")
        for router in out[label]:
            for cell, v in out[label][router].items():
                m = v["mcnemar"]
                if "p_value" not in m:
                    continue
                print(f"  {router:22s} {cell:28s} "
                      f"declared {100*v['declared_rate']['point']:5.1f}% -> "
                      f"arm {100*v['arm']['point']:5.1f}%  "
                      f"n={m['n_paired']:4d} p_holm={m.get('p_holm', float('nan')):.2e}")
        print()

    if "backbone_transfer" in out:
        print("backbone transfer (Haiku -> Sonnet), over-reach")
        for router, v in out["backbone_transfer"].items():
            m = v["mcnemar"]
            print(f"  {router:22s} {100*v['haiku']['point']:5.2f}% -> "
                  f"{100*v['sonnet']['point']:5.2f}%  "
                  f"sonnet_only={m['a_only']:3d} haiku_only={m['b_only']:3d} "
                  f"p_holm={m.get('p_holm', float('nan')):.2e}")

    u = out["usefulness_interval"]["R6_minus_R4"]
    print(f"\nusefulness, R6 minus R4 (paired, n={u['n']}):")
    print(f"  mean {u['mean_pp']:+.2f} pp, 95% CI [{u['ci_low_pp']:+.2f}, {u['ci_high_pp']:+.2f}] pp")
    print(f"  -> excludes a loss larger than {abs(u['ci_low_pp']):.1f} pp; does NOT establish equivalence")
    print(f"\nwrote {RESULTS}/cross_arm.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
