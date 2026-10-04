#!/usr/bin/env python3
"""Emit the manuscript's figures as standalone LaTeX, data drawn from results.

Three figures, each a `standalone` document compiled by tectonic to PDF and then
included by paper.tex -- the pattern used in papers_v3/DARE/figs.

  FigArchitecture       the routing pipeline: one principal-independent
                        extraction, then per-principal entitlement resolution.
  FigOverReach          over-reach by router and condition. Shows the 100 percent
                        wall that every permission-blind router hits on
                        out-of-scope requests.
  FigPolicySensitivity  the headline. R4 vs R6, declared vs shuffled policy,
                        with Wilson intervals.

Run:
  ./.venv/bin/python make_figures.py --out "../../papers_v3/Entitlement-Aware Routing for Cross-Domain Enterprise AI Agents"
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from common import RESULTS

SHORT = {
    "R1_keyword": "R1",
    "R2_embedding": "R2",
    "R3_llm_blind": "R3",
    "R4_llm_entitled": "R4",
    "R5_semantic_graph": "R5",
    "R6_entitlement_aware": "R6",
    "R7_confidence_gated": "R7",
}
CONDS = ("A_unrestricted", "B_natural", "C_foreign")
COND_LABEL = {"A_unrestricted": "unrestricted", "B_natural": "natural", "C_foreign": "foreign"}

PREAMBLE = r"""\documentclass[preview,border=3pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usetikzlibrary{positioning,arrows.meta,shapes.geometric}
\begin{document}
"""
POSTAMBLE = "\n\\end{document}\n"


def fig_architecture() -> str:
    return PREAMBLE + r"""
\begin{tikzpicture}[
  font=\footnotesize,
  box/.style={draw,rounded corners=2pt,align=center,inner sep=3pt,
              minimum height=7mm,text width=24mm},
  dec/.style={draw,diamond,aspect=2.6,align=center,inner sep=0pt,
              fill=black!5,text width=15mm,font=\scriptsize},
  lbl/.style={font=\scriptsize,align=center},
  >={Stealth[length=2mm]},
]
\node[box,fill=black!5] (q) {query $q$ \\ + data source};
\node[box,fill=black!5,right=7mm of q] (ex) {capability\\extraction};
\node[lbl,above=1.5mm of ex] {one model call per query\\(principal-independent)};

\node[box,fill=black!12,below=14mm of q] (res) {entitlement\\resolution};
\node[dec,right=7mm of res] (scope) {scope ok?};
\node[dec,right=7mm of scope] (tool) {tools held?};
\node[box,right=7mm of tool] (route) {execute route};
\node[box,below=7mm of scope] (abst) {abstain / escalate};
\node[lbl,below=1.5mm of res] {per principal,\\no model call};

\draw[->] (q) -- (ex);
\draw[->] (ex.south) .. controls +(0,-9mm) and +(0,9mm) .. (res.north);
\draw[->] (res) -- (scope);
\draw[->] (scope) -- node[above,font=\scriptsize] {yes} (tool);
\draw[->] (tool) -- node[above,font=\scriptsize] {yes} (route);
\draw[->] (scope) -- node[left,font=\scriptsize,pos=0.4] {no} (abst.north);
\draw[->] (tool.south) |- node[above,font=\scriptsize,pos=0.72] {no} (abst.east);
\end{tikzpicture}
""" + POSTAMBLE


def fig_over_reach(main: dict) -> str:
    bars = []
    for cond in CONDS:
        coords = " ".join(
            f"({SHORT[r]},{100 * main['routers'][r][cond]['over_reach']['point']:.2f})"
            for r in SHORT
        )
        bars.append(
            f"\\addplot+[ybar] coordinates {{{coords}}};\n"
            f"\\addlegendentry{{{COND_LABEL[cond]}}}"
        )
    body = "\n".join(bars)
    return PREAMBLE + rf"""
\begin{{tikzpicture}}
\begin{{axis}}[
  width=11cm, height=5.2cm,
  ybar, bar width=4.4pt,
  ymin=0, ymax=105,
  ylabel={{over-reach (\%)}},
  symbolic x coords={{R1,R2,R3,R4,R5,R6,R7}},
  xtick=data,
  ymajorgrids, grid style={{black!12}},
  legend style={{at={{(0.5,1.03)}}, anchor=south, legend columns=3, draw=none,
                 font=\footnotesize}},
  tick label style={{font=\footnotesize}},
  label style={{font=\footnotesize}},
]
{body}
\end{{axis}}
\end{{tikzpicture}}
""" + POSTAMBLE


def fig_policy(policy: dict) -> str:
    rows = []
    for router in ("R4_llm_entitled", "R6_entitlement_aware"):
        for arm in ("declared", "shuffled"):
            d = policy["over_reach_by_stratum"][router]["INFEASIBLE_SCOPE"][arm]
            f = policy["false_abstention_on_feasible"][router][arm]
            rows.append((f"{SHORT[router]} {arm}", d, f))

    def series(idx: int) -> str:
        """Coordinates with explicit asymmetric Wilson error values."""
        out = []
        for label, d, f in rows:
            src = d if idx == 0 else f
            lo = 100 * (src["point"] - src["ci_low"])
            hi = 100 * (src["ci_high"] - src["point"])
            out.append(
                f"({label},{100*src['point']:.2f}) -= (0,{lo:.2f}) += (0,{hi:.2f})"
            )
        return " ".join(out)

    labels = ",".join(r[0] for r in rows)
    return PREAMBLE + rf"""
\begin{{tikzpicture}}
\begin{{axis}}[
  width=11cm, height=5.4cm,
  ybar, bar width=9pt,
  ymin=0, ymax=45,
  ylabel={{rate (\%)}},
  symbolic x coords={{{labels}}},
  xtick=data,
  x tick label style={{font=\scriptsize, rotate=20, anchor=east}},
  ymajorgrids, grid style={{black!12}},
  legend style={{at={{(0.5,1.03)}}, anchor=south, legend columns=2, draw=none,
                 font=\footnotesize}},
  tick label style={{font=\footnotesize}},
  label style={{font=\footnotesize}},
]
\addplot+[ybar,error bars/.cd,y dir=both,y explicit] coordinates {{{series(0)}}};
\addlegendentry{{scope over-reach}}
\addplot+[ybar,error bars/.cd,y dir=both,y explicit] coordinates {{{series(1)}}};
\addlegendentry{{false abstention (feasible)}}
\end{{axis}}
\end{{tikzpicture}}
""" + POSTAMBLE


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args()

    figs = Path(args.out).resolve() / "figs"
    figs.mkdir(parents=True, exist_ok=True)

    main_a = json.loads((RESULTS / "analysis_haiku.json").read_text())
    policy = json.loads((RESULTS / "policy_sensitivity.json").read_text())

    sources = {
        "FigArchitecture": fig_architecture(),
        "FigOverReach": fig_over_reach(main_a),
        "FigPolicySensitivity": fig_policy(policy),
    }
    for name, body in sources.items():
        (figs / f"{name}.tex").write_text(body)
        print(f"wrote {figs/name}.tex")

    if args.no_compile:
        return 0

    failed = []
    for name in sources:
        proc = subprocess.run(
            ["tectonic", "-X", "compile", f"{name}.tex"],
            cwd=figs, capture_output=True, text=True, timeout=600,
        )
        if proc.returncode != 0 or not (figs / f"{name}.pdf").exists():
            failed.append(name)
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-6:]
            print(f"  FAILED {name}:\n    " + "\n    ".join(tail))
        else:
            print(f"  compiled {name}.pdf")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
