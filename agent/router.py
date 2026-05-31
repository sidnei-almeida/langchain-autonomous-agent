"""Intent classification — LLM with heuristic fallback."""
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
    "calculation",
    "comparative_research",
    "mixed_research",
    "general_chat",
)

CLASSIFIER_PROMPT = """Classify the user query for a scientific research agent.
Return ONLY valid JSON with keys:
- intent: one of {intents}
- tools_required: array from ["arxiv","wikipedia","web_search","calculator"]
- needs_clarification: boolean
- clarification_question: string (empty if not needed)
- research_depth: one of "quick","standard","deep"
- query_rewrite: clearer search-friendly version of the query

User query: {query}
"""

RECENT_KEYWORDS = ("latest", "recent", "current", "new", "2024", "2025", "2026")


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
        )

    paper_kw = (
        "paper", "papers", "arxiv", "artigo", "research paper",
        "scientific literature", "find studies",
    )
    if any(kw in q for kw in paper_kw):
        tools = ["arxiv"]
        if any(kw in q for kw in RECENT_KEYWORDS):
            tools.append("web_search")
        topic = extract_paper_topic(query) or query
        return IntentResult(
            intent="mixed_research" if len(tools) > 1 else "paper_search",
            tools_required=tools,
            research_depth="deep" if depth == "deep" else depth,
            query_rewrite=topic,
        )

    compare_kw = ("compare", "versus", "vs ", "difference between", "contrast")
    if any(kw in q for kw in compare_kw):
        return IntentResult(
            intent="comparative_research",
            tools_required=["web_search", "arxiv", "wikipedia"],
            research_depth=depth,
            query_rewrite=query,
        )

    if any(kw in q for kw in RECENT_KEYWORDS):
        tools = ["web_search"]
        if any(kw in q for kw in ("agent", "rag", "llm", "ai", "model", "research")):
            tools.append("arxiv")
        return IntentResult(
            intent="mixed_research" if "arxiv" in tools else "web_research",
            tools_required=tools,
            research_depth=depth,
            query_rewrite=query,
        )

    explain_kw = ("what is", "who is", "explain", "define", "tell me about", "how does")
    if any(kw in q for kw in explain_kw):
        return IntentResult(
            intent="concept_explanation",
            tools_required=["wikipedia"],
            research_depth=depth,
            query_rewrite=query,
        )

    return IntentResult(
        intent="general_chat",
        tools_required=[],
        research_depth=depth,
        query_rewrite=query,
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
    """LLM classification with safe heuristic fallback."""
    prompt = CLASSIFIER_PROMPT.format(
        intents=", ".join(INTENTS),
        query=query,
    )
    try:
        response = llm.invoke([
            SystemMessage(content="You classify scientific research queries. Return JSON only."),
            HumanMessage(content=prompt),
        ])
        content = response.content if hasattr(response, "content") else str(response)
        data = _parse_classifier_json(content)
        if data:
            intent = data.get("intent", "general_chat")
            if intent not in INTENTS:
                intent = "general_chat"
            tools = [
                t for t in data.get("tools_required", [])
                if t in ("arxiv", "wikipedia", "web_search", "calculator")
            ]
            rd = data.get("research_depth", depth)
            if rd not in ("quick", "standard", "deep"):
                rd = depth
            rewrite = str(data.get("query_rewrite") or query).strip()
            extracted = extract_paper_topic(query)
            if extracted and is_valid_paper_topic(extracted):
                rewrite = extracted
            elif not is_valid_paper_topic(rewrite):
                rewrite = extracted or query

            return IntentResult(
                intent=intent,
                tools_required=tools,
                needs_clarification=bool(data.get("needs_clarification", False)),
                clarification_question=str(data.get("clarification_question") or ""),
                research_depth=rd,
                query_rewrite=rewrite,
            )
    except Exception as exc:
        print(f"Intent classifier LLM failed, using heuristics: {exc}")

    return _heuristic_classify(query, depth)
