"""Shared loading and normalisation for the FDABench labelled subset.

Both characterize.py and build_benchmark.py import from here so that tool and
database normalisation cannot drift between the characterisation the paper
quotes and the benchmark the paper evaluates on.

Base benchmark: FDABench, arXiv:2509.02473 (KDD'26), MIT licence,
HuggingFace `FDAbench2026/FDAbench-Full`.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pandas as pd


def _ensure_ca_bundle() -> None:
    """Trust the macOS system roots as well as certifi's.

    This network performs TLS interception, so huggingface.co presents a
    corporate-signed chain that certifi alone rejects with
    CERTIFICATE_VERIFY_FAILED. curl works because it uses the system store.
    We build a combined bundle rather than disabling verification.
    No-op off macOS or when the caller has already set SSL_CERT_FILE.
    """
    if os.environ.get("EAR_BUNDLE_READY"):
        return
    bundle = Path(__file__).resolve().parent / "certs" / "ca-bundle.pem"
    if not bundle.exists():
        try:
            import certifi

            bundle.parent.mkdir(exist_ok=True)
            pem = Path(certifi.where()).read_bytes()
            for store in (
                "/System/Library/Keychains/SystemRootCertificates.keychain",
                "/Library/Keychains/System.keychain",
            ):
                try:
                    pem += subprocess.run(
                        ["security", "find-certificate", "-a", "-p", store],
                        capture_output=True, timeout=60,
                    ).stdout
                except (OSError, subprocess.SubprocessError):
                    pass
            bundle.write_bytes(pem)
        except Exception:
            return
    for var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        os.environ.setdefault(var, str(bundle))
    os.environ["EAR_BUNDLE_READY"] = "1"


_ensure_ca_bundle()

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
CONFIGS = ("report", "multiple", "single")

# --- Known defects in the base benchmark -----------------------------------
# Documented in DAY1_FINDINGS.md. Each is normalised here and counted, never
# silently dropped.
#
# (1) The SQL-generation tool ships under two names, split by config:
#     `generate_sql` appears only in the report config.
TOOL_ALIASES = {"generate_sql": "generated_sql"}
#
# (2) `san_francisco_plus` and `SAN_FRANCISCO_PLUS` are one database. Database
#     identity is therefore case-folded everywhere.
#
# (3) On some tasks the database named inside the gold route differs from the
#     task's own `db` column. The gold route is authoritative, because it is
#     what the route actually executes against.

# Tools whose use is a governance-relevant capability rather than a plain
# compute step. These form the tool-class entitlement axis.
GOVERNED_TOOLS = ("web_search", "vector_search", "file_system")


def canon_db(name: str | None) -> str | None:
    """Case-fold a database identifier. See defect (2)."""
    return name.lower() if name else None


def canon_tool(name: str | None) -> str | None:
    """Resolve the SQL-generation alias. See defect (1)."""
    return TOOL_ALIASES.get(name, name) if name else None


def parse_route(arr) -> list[dict]:
    """gold_subtasks ships as an ndarray of JSON strings."""
    out = []
    for item in arr:
        if isinstance(item, str):
            try:
                item = json.loads(item)
            except json.JSONDecodeError:
                continue
        if isinstance(item, dict):
            out.append(item)
    return out


def route_tools(route: list[dict]) -> list[str]:
    """Ordered, alias-normalised tool sequence of a gold route."""
    return [canon_tool(s["tool"]) for s in route if s.get("tool")]


def route_database(route: list[dict]) -> str | None:
    """The single database a gold route executes against, case-folded.

    Every labelled task names exactly one; verified in DAY1_FINDINGS.md.
    """
    for step in route:
        inp = step.get("input")
        if isinstance(inp, dict) and inp.get("database_name"):
            return canon_db(inp["database_name"])
    return None


def load_labelled() -> pd.DataFrame:
    """The 834 tasks that ship a gold route, with derived columns.

    Adds: config, gold_route, gold_tools, gold_tool_set, gold_db,
    db_column_disagrees, required_governed.
    """
    frames = []
    for cfg in CONFIGS:
        df = pd.read_parquet(DATA / f"{cfg}.parquet")
        df = df[df["gold_subtasks"].notna()].copy()
        df["config"] = cfg
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    routes = [parse_route(a) for a in df["gold_subtasks"]]
    df["gold_route"] = routes
    df["gold_tools"] = [route_tools(r) for r in routes]
    df["gold_tool_set"] = [tuple(sorted(set(t))) for t in df["gold_tools"]]
    df["gold_db"] = [route_database(r) for r in routes]
    df["db_column_disagrees"] = [
        canon_db(col) != gold for col, gold in zip(df["db"], df["gold_db"])
    ]
    df["required_governed"] = [
        tuple(t for t in GOVERNED_TOOLS if t in s) for s in df["gold_tool_set"]
    ]
    return df


def config_totals() -> dict[str, dict[str, int]]:
    """Total vs labelled task counts per config, for reporting the 834/2007 split."""
    out = {}
    for cfg in CONFIGS:
        full = pd.read_parquet(DATA / f"{cfg}.parquet")
        out[cfg] = {
            "total": len(full),
            "labelled": int(full["gold_subtasks"].notna().sum()),
        }
    return out
