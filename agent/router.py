"""Intent classification — LLM with heuristic fallback and tooling-aware overrides."""
from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from arxiv_search import extract_paper_topic, is_valid_paper_topic

from agent.state import IntentResult, ResearchDepth

INTENTS = (
    "concept_explanation",
    "paper_search",
    "web_research",
    "technology_discovery",
    "tool_comparison",
    "calculation",
    "comparative_research",
    "mixed_research",
    "general_chat",
)

TOOL_NAMES = ("arxiv", "wikipedia", "web_search", "calculator")

# Domain signals for RAG / vector DB / tooling questions (not academic paper search).
TOOLING_DOMAIN_KEYWORDS = (
    "faiss",
    "vector database",
    "vector databases",
    "vector store",
    "vector db",
    "vector dbs",
    "embeddings",
    "embedding model",
    "semantic search",
    "retrieval augmented",
    "retrieval-augmented",
    "chroma",
    "qdrant",
    "milvus",
    "weaviate",
    "pinecone",
    "lancedb",
    "pgvector",
    "elasticsearch vector",
    "opensearch vector",
    "vespa",
    "vectorizing",
    "vectorization",
    "vectorisation",
    "similarity search",
    "ann index",
    "hnsw",
)

TOOLING_ACTION_KEYWORDS = (
    "alternative",
    "alternatives",
    "replace",
    "replaces",
    "replacing",
    "replacement",
    "best ",
    " tool",
    "tools ",
    "library",
    "libraries",
    "framework",
    "platform",
    "database",
    "databases",
    "production",
    "deploy",
    "deployment",
    "stack",
    "new technology",
    "new stack",
)

PAPER_EXPLICIT_KEYWORDS = (
    "paper",
    "papers",
    "arxiv",
    "artigo",
    "research paper",
    "scientific literature",
    "publication",
    "publications",
    "research article",
    "peer-reviewed",
    "find studies",
    "find papers",
    "recent papers",
)

COMPARE_KEYWORDS = ("compare", "versus", " vs ", " vs.", "difference between", "contrast")

RECENT_KEYWORDS = ("latest", "recent", "current", "new", "2024", "2025", "2026")

EXPLAIN_KEYWORDS = ("what is", "who is", "explain", "define", "tell me about", "how does")

CLASSIFIER_PROMPT = """Classify the user query for a research agent (science + technology tooling).

Return ONLY valid JSON with keys:
- intent: one of {intents}
- tools_required: array from ["arxiv","wikipedia","web_search","calculator"]
- tools_optional: array (same values) — tools that may run only as complement
- needs_clarification: boolean
- clarification_question: string (empty if not needed)
- research_depth: one of "quick","standard","deep"
- query_rewrite: clearer search-friendly version of the query
- reason: short string explaining the routing choice

Routing rules (critical):
1. Questions about tools, libraries, platforms, frameworks, vector databases, or alternatives
   (FAISS, Qdrant, Milvus, RAG stacks, embeddings, production deploy) are NOT paper_search.
   Use intent "technology_discovery" or "tool_comparison" with tools_required including "web_search".
   Put "arxiv" in tools_optional only, never as the only required tool, unless the user explicitly asks for papers.

2. Use "paper_search" or require "arxiv" only when the user asks for papers, arXiv, studies,
   scientific literature, publications, or benchmarks from research articles.

3. "What is X" / "explain" / "define" → concept_explanation; prefer wikipedia optional, not required arxiv.

4. "latest/new/current" about **tools or technologies** → web_search first, not arxiv first.

5. Comparisons like "Qdrant vs Milvus" → tool_comparison + web_search.

User query: {query}
"""


def is_explicit_paper_request(query: str) -> bool:
    q = query.lower()
    if any(kw in q for kw in PAPER_EXPLICIT_KEYWORDS):
        return True
    if extract_paper_topic(query):
        return True
    return False


def _has_rag_keyword(query: str) -> bool:
    return bool(re.search(r"\brag\b", query, re.IGNORECASE))


def is_tooling_domain_query(query: str) -> bool:
    q = query.lower()
    if _has_rag_keyword(query):
        return True
    if any(kw in q for kw in TOOLING_DOMAIN_KEYWORDS):
        return True
    if "vector" in q and any(
        w in q for w in ("database", "store", "db", "search", "index")
    ):
        return True
    return False


def is_tooling_action_query(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in TOOLING_ACTION_KEYWORDS)


def is_tooling_query(query: str) -> bool:
    """Technology discovery / vector DB / RAG stack questions — web-first, not arXiv-first."""
    if not is_tooling_domain_query(query):
        return False
    q = query.lower()
    if is_tooling_action_query(query):
        return True
    if any(kw in q for kw in RECENT_KEYWORDS):
        return True
    if "technology" in q or "technologies" in q:
        return True
    return False


def is_tool_comparison_query(query: str) -> bool:
    q = query.lower()
    if not any(kw in q for kw in COMPARE_KEYWORDS):
        return False
    return is_tooling_domain_query(query) or any(
        w in q for w in ("database", "vector", "library", "framework", "tool", "stack")
    )


def is_tooling_intent(intent: str | None) -> bool:
    return intent in ("technology_discovery", "tool_comparison", "web_research")


def _normalize_tools(tools: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for t in tools:
        if t in TOOL_NAMES and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _web_first_tools(
    required: list[str],
    optional: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Ensure web_search is required first; demote arxiv to optional unless explicit papers."""
    req = _normalize_tools(required)
    opt = _normalize_tools(optional or [])

    if "arxiv" in req:
        req = [t for t in req if t != "arxiv"]
        if "arxiv" not in opt:
            opt.append("arxiv")

    if "web_search" not in req:
        req.insert(0, "web_search")

    if "wikipedia" not in req and "wikipedia" not in opt:
        opt.insert(0, "wikipedia")

    return req, opt


def apply_routing_overrides(result: IntentResult, query: str) -> IntentResult:
    """Post-process classifier/heuristic output for consistent tooling vs paper routing."""
    q = query.lower()

    if is_explicit_paper_request(query):
        tools = list(result.tools_required)
        if "arxiv" not in tools:
            tools.insert(0, "arxiv")
        if any(kw in q for kw in RECENT_KEYWORDS) and "web_search" not in tools:
            tools.append("web_search")
        topic = extract_paper_topic(query) or result.query_rewrite or query
        result.intent = "mixed_research" if len(tools) > 1 else "paper_search"
        result.tools_required = _normalize_tools(tools)
        result.tools_optional = _normalize_tools(result.tools_optional)
        if is_valid_paper_topic(topic):
            result.query_rewrite = topic
        return result

    if is_tool_comparison_query(query):
        req, opt = _web_first_tools(["web_search"])
        result.intent = "tool_comparison"
        result.tools_required = req
        result.tools_optional = opt
        result.needs_clarification = False
        result.clarification_question = ""
        return result

    if is_tooling_query(query):
        req, opt = _web_first_tools(["web_search"])
        result.intent = (
            "tool_comparison"
            if is_tool_comparison_query(query)
            else "technology_discovery"
        )
        result.tools_required = req
        result.tools_optional = opt
        result.needs_clarification = False
        result.clarification_question = ""
        return result

    # Demote arxiv-only misroutes when query is clearly tooling-related
    if result.intent in ("paper_search", "mixed_research") and is_tooling_domain_query(query):
        if not is_explicit_paper_request(query):
            req, opt = _web_first_tools(result.tools_required, result.tools_optional)
            result.intent = "technology_discovery"
            result.tools_required = req
            result.tools_optional = opt
            result.needs_clarification = False
            result.clarification_question = ""

    if result.intent in ("technology_discovery", "tool_comparison", "web_research"):
        req, opt = _web_first_tools(result.tools_required, result.tools_optional)
        result.tools_required = req
        result.tools_optional = opt

    return result


def _heuristic_classify(query: str, depth: ResearchDepth = "standard") -> IntentResult:
    q = query.lower().strip()

    if any(kw in q for kw in ("calculate", "compute", "sqrt", "sin(", "cos(")) or (
        re.search(r"[\d+\-*/^=]", q) and any(c in q for c in "+-*/=")
    ):
        return IntentResult(
            intent="calculation",
            tools_required=["calculator"],
            research_depth=depth,
            query_rewrite=query,
            reason="Mathematical expression detected.",
        )

    if is_explicit_paper_request(query):
        tools = ["arxiv"]
        if any(kw in q for kw in RECENT_KEYWORDS):
            tools.append("web_search")
        topic = extract_paper_topic(query) or query
        return apply_routing_overrides(
            IntentResult(
                intent="mixed_research" if len(tools) > 1 else "paper_search",
                tools_required=tools,
                research_depth="deep" if depth == "deep" else depth,
                query_rewrite=topic,
                reason="Explicit paper or arXiv request.",
            ),
            query,
        )

    if any(kw in q for kw in EXPLAIN_KEYWORDS) and not is_tool_comparison_query(query):
        return apply_routing_overrides(
            IntentResult(
                intent="concept_explanation",
                tools_required=["wikipedia"],
                tools_optional=["web_search"] if is_tooling_domain_query(query) else [],
                research_depth=depth,
                query_rewrite=query,
                reason="Explanation or definition request.",
            ),
            query,
        )

    if is_tool_comparison_query(query):
        req, opt = _web_first_tools(["web_search"])
        return IntentResult(
            intent="tool_comparison",
            tools_required=req,
            tools_optional=opt,
            research_depth=depth,
            query_rewrite=query,
            reason="Tool or platform comparison detected.",
        )

    if is_tooling_query(query):
        req, opt = _web_first_tools(["web_search"])
        return IntentResult(
            intent="technology_discovery",
            tools_required=req,
            tools_optional=opt,
            research_depth=depth,
            query_rewrite=query,
            reason="Technology or vector/RAG tooling discovery query.",
        )

    if any(kw in q for kw in COMPARE_KEYWORDS):
        return apply_routing_overrides(
            IntentResult(
                intent="comparative_research",
                tools_required=["web_search", "wikipedia"],
                tools_optional=["arxiv"],
                research_depth=depth,
                query_rewrite=query,
                reason="General comparison (non-tooling-specific).",
            ),
            query,
        )

    if any(kw in q for kw in RECENT_KEYWORDS):
        tools = ["web_search"]
        if any(
            kw in q
            for kw in ("agent", "llm", "model")
        ) and not is_tooling_domain_query(query):
            tools.append("arxiv")
        intent = "mixed_research" if "arxiv" in tools else "web_research"
        return apply_routing_overrides(
            IntentResult(
                intent=intent,
                tools_required=tools,
                research_depth=depth,
                query_rewrite=query,
                reason="Recency-focused query; web first unless academic papers requested.",
            ),
            query,
        )

    return IntentResult(
        intent="general_chat",
        tools_required=[],
        research_depth=depth,
        query_rewrite=query,
        reason="No specialized tools required.",
    )


def _parse_classifier_json(text: str) -> dict | None:
    text = text.strip()
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def classify_intent(
    llm,
    query: str,
    depth: ResearchDepth = "standard",
) -> IntentResult:
    """LLM classification with safe heuristic fallback and routing overrides."""
    prompt = CLASSIFIER_PROMPT.format(
        intents=", ".join(INTENTS),
        query=query,
    )
    try:
        response = llm.invoke([
            SystemMessage(
                content=(
                    "You classify research and technology queries. "
                    "Route vector DB / RAG tooling questions to web_search, not arXiv-only. "
                    "Return JSON only."
                )
            ),
            HumanMessage(content=prompt),
        ])
        content = response.content if hasattr(response, "content") else str(response)
        data = _parse_classifier_json(content)
        if data:
            intent = data.get("intent", "general_chat")
            if intent not in INTENTS:
                intent = "general_chat"
            tools = _normalize_tools(data.get("tools_required", []))
            optional = _normalize_tools(data.get("tools_optional", []))
            rd = data.get("research_depth", depth)
            if rd not in ("quick", "standard", "deep"):
                rd = depth
            rewrite = str(data.get("query_rewrite") or query).strip()
            if not is_tooling_query(query) and not is_explicit_paper_request(query):
                extracted = extract_paper_topic(query)
                if extracted and is_valid_paper_topic(extracted):
                    rewrite = extracted
                elif not is_valid_paper_topic(rewrite):
                    rewrite = extracted or query
            elif not is_valid_paper_topic(rewrite):
                rewrite = query

            result = IntentResult(
                intent=intent,
                tools_required=tools,
                tools_optional=optional,
                needs_clarification=bool(data.get("needs_clarification", False)),
                clarification_question=str(data.get("clarification_question") or ""),
                research_depth=rd,
                query_rewrite=rewrite,
                reason=str(data.get("reason") or ""),
            )
            return apply_routing_overrides(result, query)
    except Exception as exc:
        print(f"Intent classifier LLM failed, using heuristics: {exc}")

    return apply_routing_overrides(_heuristic_classify(query, depth), query)
