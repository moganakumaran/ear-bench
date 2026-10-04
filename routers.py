"""The seven routers compared in the paper.

Routing decision
----------------
FDABench supplies the database as task input rather than asking the agent to
discover it, so a route here is: WHICH CAPABILITIES to invoke against a named
data source, and WHETHER TO PROCEED AT ALL. Requiring routers to also guess the
database from 73 names would add a sub-task the paper is not about.

Every gold route contains the SQL core (get_schema_info, generated_sql,
execute_sql), so the live decision is over the three governed tools plus
abstention.

Permission-blind vs entitlement-aware
-------------------------------------
R1, R2, R3 and R5 never see entitlements, so their decision cannot depend on the
principal: they are evaluated ONCE PER TASK and the same decision is scored
against all three conditions. R4 sees entitlements in its prompt and must be
called once per (task, principal).

R6 splits the work: ONE principal-independent LLM extraction per task, then
deterministic entitlement-aware scoring per principal. That is the proposed
method's efficiency claim — one model call per query regardless of how many
principals ask it — and it is measured, not asserted.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from llm import extract_json
from metrics import RouteDecision

SQL_CORE = frozenset({"get_schema_info", "generated_sql", "execute_sql"})
GOVERNED = ("web_search", "vector_search", "file_system")

TOOL_DESCRIPTIONS = {
    "get_schema_info": "Return the schema of the named database.",
    "generated_sql": "Generate SQL for the question against that schema.",
    "execute_sql": "Execute the generated SQL and return rows.",
    "web_search": "Search the public web. Sends query content outside the platform.",
    "vector_search": "Search the internal document corpus (reports, PDFs, papers).",
    "file_system": "Read files directly from the filesystem.",
}


# ---------------------------------------------------------------------------
# R1 - keyword / rule router
# ---------------------------------------------------------------------------
# Cues are lexical only. Written against the tool descriptions, not tuned on
# benchmark labels; tuning them on gold would make R1 an oracle in disguise.
_WEB_CUES = re.compile(
    r"\b(latest|current|recent|news|today|announce\w*|market|public(ly)?|online|"
    r"industry|report(ed|s)? (by|in) the (press|media)|trend\w*)\b",
    re.I,
)
_DOC_CUES = re.compile(
    r"\b(stud\w+|paper\w*|pdf|document\w*|literature|research|report\w*|"
    r"publication\w*|article\w*|whitepaper|\.pdf)\b",
    re.I,
)
_FS_CUES = re.compile(r"\b(file|folder|directory|csv|spreadsheet|attachment|upload\w*)\b", re.I)


def r1_keyword(query: str) -> RouteDecision:
    tools = set(SQL_CORE)
    if _WEB_CUES.search(query):
        tools.add("web_search")
    if _DOC_CUES.search(query):
        tools.add("vector_search")
    if _FS_CUES.search(query):
        tools.add("file_system")
    return RouteDecision(tools=frozenset(tools), rationale="keyword")


# ---------------------------------------------------------------------------
# R2 - dense embedding similarity router
# ---------------------------------------------------------------------------
class EmbeddingRouter:
    """Cosine similarity between the query and each governed tool's description.

    Anthropic exposes no embedding endpoint, so this runs a local
    sentence-transformers model. Thresholds are calibrated once on a held-out
    slice (see run_routers.py --calibrate), never on the evaluation set.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self._tool_vecs = self.model.encode(
            [TOOL_DESCRIPTIONS[t] for t in GOVERNED], normalize_embeddings=True
        )
        self.thresholds = dict.fromkeys(GOVERNED, 0.25)

    def scores(self, queries: list[str]):
        import numpy as np

        qv = self.model.encode(queries, normalize_embeddings=True, batch_size=64)
        return np.asarray(qv) @ np.asarray(self._tool_vecs).T

    def decide(self, score_row) -> RouteDecision:
        tools = set(SQL_CORE)
        for i, tool in enumerate(GOVERNED):
            if score_row[i] >= self.thresholds[tool]:
                tools.add(tool)
        return RouteDecision(tools=frozenset(tools), rationale="embedding")


# ---------------------------------------------------------------------------
# R5 - semantic / knowledge-graph router (AgentRouter-style, permission-blind)
# ---------------------------------------------------------------------------
class SemanticGraphRouter:
    """Scores governed tools by propagating over a query-entity / database /
    domain graph, in the spirit of KG-guided routing (AgentRouter,
    arXiv:2510.05445). It consults the SAME enterprise graph the proposed method
    uses, but has no entitlement layer — isolating entitlement awareness as the
    only difference between R5 and R6.

    Edge weights are the empirical rate at which each domain's tasks require
    each governed tool, estimated on the CALIBRATION SPLIT only.
    """

    def __init__(self, domain_tool_rates: dict[str, dict[str, float]], threshold: float = 0.5):
        self.rates = domain_tool_rates
        self.threshold = threshold

    def decide(self, data_domain: str) -> RouteDecision:
        tools = set(SQL_CORE)
        rates = self.rates.get(data_domain, {})
        for tool in GOVERNED:
            if rates.get(tool, 0.0) >= self.threshold:
                tools.add(tool)
        return RouteDecision(tools=frozenset(tools), rationale=f"graph:{data_domain}")


# ---------------------------------------------------------------------------
# LLM routers
# ---------------------------------------------------------------------------
_CATALOGUE = "\n".join(f"- {t}: {d}" for t, d in TOOL_DESCRIPTIONS.items())

R3_SYSTEM = (
    "You route analytical questions to data tools. Choose the minimal set of "
    "tools needed to answer the question correctly. Reply with JSON only."
)

R3_PROMPT = """Question: {query}
Database available: {db}

Tools:
{catalogue}

get_schema_info, generated_sql and execute_sql are always required to query the
database. Decide which of web_search, vector_search, file_system are also needed.

Reply with JSON only:
{{"tools": ["..."], "reason": "<one short sentence>"}}"""

R4_SYSTEM = (
    "You route analytical questions to data tools on behalf of a specific "
    "principal. The principal's entitlements are binding: you must never select "
    "a tool they do not hold, and never query a database outside their data "
    "scope. If the question cannot be answered within their entitlements, "
    "abstain. Reply with JSON only."
)

R4_PROMPT = """Question: {query}
Database this question requires: {db}

Tools:
{catalogue}

PRINCIPAL: {principal}
Databases this principal may query: {entitled_dbs}
Governed tools this principal holds: {entitled_tools}
(get_schema_info, generated_sql, execute_sql are available to everyone, but only
against a database inside the principal's data scope.)

Decide. If the required database is outside the principal's scope, or the
question needs a governed tool the principal does not hold, you must abstain.

Reply with JSON only:
{{"abstain": true|false, "tools": ["..."], "reason": "<one short sentence>"}}"""

R6_SYSTEM = (
    "You analyse an analytical question and report which information sources it "
    "genuinely requires. You do not make access decisions. Reply with JSON only."
)

R6_PROMPT = """Question: {query}
Database this question concerns: {db}

Report which of these the question genuinely requires, independent of who is asking:
- web_search: needs public/web information beyond the database
- vector_search: needs internal documents, studies, reports or PDFs
- file_system: needs direct file access

Reply with JSON only:
{{"needs_web_search": true|false,
  "needs_vector_search": true|false,
  "needs_file_system": true|false,
  "confidence": 0.0-1.0,
  "reason": "<one short sentence>"}}"""


def _tools_from_list(raw) -> frozenset[str]:
    if not isinstance(raw, list):
        return frozenset(SQL_CORE)
    picked = {t for t in raw if isinstance(t, str) and t in TOOL_DESCRIPTIONS}
    return frozenset(picked | SQL_CORE)


@dataclass
class RouterError(Exception):
    """The router produced unparseable output. Recorded, never coerced to abstain."""

    instance_id: str
    raw: str


def r3_llm(llm, query: str, db: str, seed: int) -> tuple[RouteDecision, bool]:
    text = llm.complete(
        R3_PROMPT.format(query=query, db=db, catalogue=_CATALOGUE),
        system=R3_SYSTEM,
        seed=seed,
    )
    data = extract_json(text)
    if data is None:
        return RouteDecision(tools=frozenset(SQL_CORE), rationale="parse-error"), True
    return (
        RouteDecision(tools=_tools_from_list(data.get("tools")), rationale=str(data.get("reason", ""))[:200]),
        False,
    )


def r4_llm(llm, instance: dict, entitled_dbs: list[str], seed: int) -> tuple[RouteDecision, bool]:
    prompt = R4_PROMPT.format(
        query=instance["query"],
        db=instance["gold_db"],
        catalogue=_CATALOGUE,
        principal=instance["principal"],
        entitled_dbs=", ".join(entitled_dbs) if entitled_dbs else "(none)",
        entitled_tools=", ".join(instance["entitled_governed_tools"]) or "(none)",
    )
    text = llm.complete(prompt, system=R4_SYSTEM, seed=seed)
    data = extract_json(text)
    if data is None:
        return RouteDecision(tools=frozenset(SQL_CORE), rationale="parse-error"), True
    if data.get("abstain") is True:
        return RouteDecision.abstain(str(data.get("reason", ""))[:200]), False
    return (
        RouteDecision(
            tools=_tools_from_list(data.get("tools")),
            database=instance["gold_db"],
            rationale=str(data.get("reason", ""))[:200],
        ),
        False,
    )


def r6_extract(llm, query: str, db: str, seed: int) -> tuple[dict | None, bool]:
    """ONE principal-independent call per task. Reused for every principal."""
    text = llm.complete(
        R6_PROMPT.format(query=query, db=db), system=R6_SYSTEM, seed=seed
    )
    data = extract_json(text)
    return (data, data is None)


def r6_decide(need: dict | None, instance: dict) -> RouteDecision:
    """Deterministic entitlement-aware scoring. No model call.

    Hard filter first: a principal outside the data scope has no legal route at
    all, so the decision is abstention regardless of what the query needs.
    Then feasibility: if the query needs a governed tool the principal does not
    hold, the gold route cannot be completed, so abstain rather than silently
    returning a degraded answer.
    """
    if not instance["database_entitled"]:
        return RouteDecision.abstain("database outside principal data scope")

    needed = set()
    if need:
        for tool in GOVERNED:
            if need.get(f"needs_{tool}") is True:
                needed.add(tool)

    held = set(instance["entitled_governed_tools"])
    missing = needed - held
    if missing:
        return RouteDecision.abstain(f"requires unheld tool(s): {sorted(missing)}")
    return RouteDecision(
        tools=frozenset(SQL_CORE | needed),
        database=instance["gold_db"],
        rationale="entitled",
    )


def r7_decide(need: dict | None, instance: dict, min_confidence: float) -> RouteDecision:
    """R6 plus a confidence gate.

    R6's abstentions are entitlement-driven and certain; this adds abstention
    when the extraction itself is unreliable, so a low-confidence read of what
    the query needs does not silently become a tool selection.
    """
    base = r6_decide(need, instance)
    if base.abstained:
        return base
    conf = (need or {}).get("confidence")
    try:
        conf = float(conf)
    except (TypeError, ValueError):
        conf = 0.0
    if conf < min_confidence:
        return RouteDecision.abstain(f"low extraction confidence {conf:.2f}")
    return base
