"""Statistics for the routing results.

Uncertainty here is over the sample of tasks, not over decode noise, so rates
carry Wilson intervals computed over instances and router pairs are compared
with McNemar on the SAME instances. Both are standard for paired binary
outcomes and neither assumes normality at the small counts we hit in the
INFEASIBLE_TOOL stratum.
"""
from __future__ import annotations

from dataclasses import dataclass

from scipy import stats as sps


@dataclass
class Interval:
    point: float
    low: float
    high: float
    n: int

    def tex(self, pct: bool = True) -> str:
        if pct:
            return f"{100*self.point:.1f} [{100*self.low:.1f}, {100*self.high:.1f}]"
        return f"{self.point:.3f} [{self.low:.3f}, {self.high:.3f}]"

    def as_dict(self) -> dict:
        return {
            "point": round(self.point, 4),
            "ci_low": round(self.low, 4),
            "ci_high": round(self.high, 4),
            "n": self.n,
        }


def wilson(successes: int, n: int, confidence: float = 0.95) -> Interval:
    """Wilson score interval. Correct at proportions near 0 and 1, where the
    normal approximation gives intervals that run outside [0, 1] -- which
    matters here because several routers sit at exactly 0 over-reach."""
    if n == 0:
        return Interval(0.0, 0.0, 0.0, 0)
    z = sps.norm.ppf(1 - (1 - confidence) / 2)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return Interval(p, max(0.0, centre - half), min(1.0, centre + half), n)


def rate_ci(rows: list[dict], key: str, confidence: float = 0.95) -> Interval:
    usable = [r for r in rows if r.get(key) is not None]
    return wilson(sum(bool(r[key]) for r in usable), len(usable), confidence)


def mcnemar(a_rows: list[dict], b_rows: list[dict], key: str) -> dict:
    """Exact McNemar on paired per-instance binary outcomes.

    `a_rows` and `b_rows` must cover the same instances. Uses the exact
    binomial test rather than the chi-square approximation, because the
    discordant counts are small when one router is at or near zero.
    """
    a = {r["instance_id"]: r for r in a_rows}
    b = {r["instance_id"]: r for r in b_rows}
    shared = sorted(set(a) & set(b))
    if not shared:
        return {"error": "no shared instances"}

    a_only = b_only = both = neither = 0
    for iid in shared:
        av, bv = a[iid].get(key), b[iid].get(key)
        if av is None or bv is None:
            continue
        av, bv = bool(av), bool(bv)
        if av and bv:
            both += 1
        elif av and not bv:
            a_only += 1
        elif bv and not av:
            b_only += 1
        else:
            neither += 1

    discordant = a_only + b_only
    if discordant == 0:
        return {
            "n_paired": len(shared),
            "a_only": 0, "b_only": 0, "both": both, "neither": neither,
            "p_value": 1.0,
            "note": "no discordant pairs; the routers never differ on this outcome",
        }
    p = sps.binomtest(a_only, discordant, 0.5).pvalue
    return {
        "n_paired": len(shared),
        "a_only": a_only,
        "b_only": b_only,
        "both": both,
        "neither": neither,
        "discordant": discordant,
        "p_value": float(p),
    }


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjustment across a family of comparisons."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[str, float] = {}
    running = 0.0
    for i, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[name] = round(running, 6)
    return adjusted


if __name__ == "__main__":
    # Sanity: a proportion at exactly zero must still get a usable interval.
    z = wilson(0, 200)
    assert z.point == 0.0 and z.low < 1e-9 and 0 < z.high < 0.05, z
    half = wilson(100, 200)
    assert abs(half.point - 0.5) < 1e-9 and half.low < 0.5 < half.high

    a = [{"instance_id": str(i), "x": i < 10} for i in range(100)]
    b = [{"instance_id": str(i), "x": i < 50} for i in range(100)]
    r = mcnemar(a, b, "x")
    assert r["b_only"] == 40 and r["a_only"] == 0 and r["p_value"] < 1e-9, r

    same = mcnemar(a, a, "x")
    assert same["p_value"] == 1.0 and "note" in same

    adj = holm({"a": 0.01, "b": 0.04, "c": 0.5})
    assert adj["a"] <= adj["b"] <= adj["c"], adj
    print("stats self-tests passed")
