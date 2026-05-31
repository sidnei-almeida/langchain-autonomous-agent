"""Tool implementations and multi-tool execution."""
from __future__ import annotations

import math
import re
from typing import Any

from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from arxiv_search import (
    format_structured_result_for_agent,
    resolve_arxiv_topic,
    search_scientific_papers_structured,
    wants_recent_papers,
)
from agent.router import is_explicit_paper_request, is_tooling_intent
from agent.state import EvidenceItem, ResearchState

RECENT_KEYWORDS = ("latest", "recent", "current", "new", "2024", "2025", "2026")

# Execution order: web/wikipedia before arXiv unless explicit paper search.
TOOL_ORDER_PAPER_FIRST = ("arxiv", "web_search", "wikipedia", "calculator")
TOOL_ORDER_WEB_FIRST = ("web_search", "wikipedia", "arxiv", "calculator")


def _tool_execution_order(state: ResearchState) -> tuple[str, ...]:
    intent = state.intent.intent if state.intent else "general_chat"
    required = list(state.intent.tools_required) if state.intent else []
    if intent == "paper_search" or (
        "arxiv" in required and is_explicit_paper_request(state.user_query)
    ):
        base = TOOL_ORDER_PAPER_FIRST
    else:
        base = TOOL_ORDER_WEB_FIRST

    ordered = [t for t in base if t in required]
    for tool in required:
        if tool not in ordered:
            ordered.append(tool)
    return tuple(ordered)


def _arxiv_weak_or_clarification(structured: dict[str, Any]) -> bool:
    if structured.get("type") == "clarification":
        return True
    papers = structured.get("papers") or []
    return structured.get("type") == "paper_results" and len(papers) == 0


def _should_continue_after_weak_arxiv(state: ResearchState) -> bool:
    if not state.intent:
        return False
    if is_tooling_intent(state.intent.intent):
        return True
    if state.intent.intent == "concept_explanation":
        return True
    if not is_explicit_paper_request(state.user_query):
        return True
    return False


class ToolRegistry:
    def __init__(self):
        self._web = DuckDuckGoSearchRun()
        self._wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

    def run_calculator(self, query: str) -> tuple[str, EvidenceItem | None]:
        expr = _extract_calc_expression(query) or query
        result = calculator(expr)
        evidence = EvidenceItem(
            title=f"Calculation: {expr}",
            url="",
            source_type="calculation",
            snippet=result,
            relevance_score=1.0,
            metadata={"expression": expr, "result": result},
        )
        return result, evidence

    def run_wikipedia(self, query: str) -> tuple[str, list[EvidenceItem]]:
        text = self._wiki.run(query)
        evidence = [
            EvidenceItem(
                title=f"Wikipedia: {query[:80]}",
                url="https://en.wikipedia.org/wiki/Special:Search?search="
                + query.replace(" ", "+"),
                source_type="wikipedia",
                snippet=text[:800],
                relevance_score=0.75,
            )
        ]
        return text, evidence

    def run_web(self, query: str) -> tuple[str, list[EvidenceItem]]:
        text = self._web.run(query)
        items: list[EvidenceItem] = []
        urls = re.findall(r"https?://[^\s\)\]]+", text)
        if urls:
            for i, url in enumerate(urls[:3]):
                items.append(
                    EvidenceItem(
                        title=f"Web result {i + 1}",
                        url=url.rstrip(".,;"),
                        source_type="web",
                        snippet=text[:400],
                        relevance_score=max(0.5, 0.85 - i * 0.1),
                    )
                )
        else:
            items.append(
                EvidenceItem(
                    title=f"Web search: {query[:60]}",
                    url="",
                    source_type="web",
                    snippet=text[:800],
                    relevance_score=0.7,
                )
            )
        return text, items

    def run_arxiv(
        self,
        query: str,
        *,
        user_query: str | None = None,
        depth: str = "standard",
        recent_only: bool = False,
    ) -> tuple[str, dict[str, Any], list[EvidenceItem]]:
        original = user_query or query
        topic = resolve_arxiv_topic(original, query)
        max_papers = 5 if depth == "deep" else 3
        structured = search_scientific_papers_structured(
            topic,
            max_papers=max_papers,
            recent_only=recent_only or wants_recent_papers(original),
        )
        formatted = format_structured_result_for_agent(structured)
        evidence: list[EvidenceItem] = []

        if structured.get("type") == "paper_results":
            for paper in structured.get("papers") or []:
                score = paper.get("relevanceScore", 0)
                evidence.append(
                    EvidenceItem(
                        title=paper.get("title", ""),
                        url=paper.get("url", ""),
                        source_type="arxiv",
                        snippet=paper.get("summary", "")[:600],
                        relevance_score=min(1.0, score / 20.0),
                        metadata={
                            "authors": paper.get("authors"),
                            "year": paper.get("year"),
                            "categories": paper.get("categories"),
                            "whyItMatches": paper.get("whyItMatches"),
                            "pdfUrl": paper.get("pdfUrl"),
                        },
                    )
                )

        return formatted, structured, evidence

    def _run_web_fallback(self, state: ResearchState, query: str) -> None:
        if "web_search" in state.tools_used:
            return
        try:
            _, evs = self.run_web(query)
            state.tools_used.append("web_search")
            state.evidence.extend(evs)
            state.limitations.append(
                "arXiv had no strong matches; supplemented with web search for tooling context."
            )
        except Exception as exc:
            print(f"Web fallback failed: {exc}")
            state.limitations.append("Web fallback search encountered an error.")

    def execute_tools(self, state: ResearchState) -> ResearchState:
        if not state.intent:
            return state

        query = state.intent.query_rewrite or state.user_query
        tools = _tool_execution_order(state)
        recent = any(kw in query.lower() for kw in RECENT_KEYWORDS)
        arxiv_weak = False

        for tool in tools:
            try:
                if tool == "calculator":
                    _, ev = self.run_calculator(query)
                    state.tools_used.append("calculator")
                    if ev:
                        state.evidence.append(ev)

                elif tool == "wikipedia":
                    _, evs = self.run_wikipedia(query)
                    state.tools_used.append("wikipedia")
                    state.evidence.extend(evs)

                elif tool == "web_search":
                    _, evs = self.run_web(query)
                    state.tools_used.append("web_search")
                    state.evidence.extend(evs)

                elif tool == "arxiv":
                    _, structured, evs = self.run_arxiv(
                        query,
                        user_query=state.user_query,
                        depth=state.depth,
                        recent_only=recent,
                    )
                    state.paper_search = structured

                    if _arxiv_weak_or_clarification(structured):
                        arxiv_weak = True
                        if _should_continue_after_weak_arxiv(state):
                            state.limitations.append(
                                "arXiv search did not return strong paper matches "
                                "(not required for tooling/technology questions)."
                            )
                            continue

                        state.tools_used.append("search_scientific_papers")
                        state.intent.needs_clarification = True
                        state.intent.clarification_question = structured.get(
                            "message", ""
                        )
                        options = structured.get("options") or []
                        if options:
                            state.intent.clarification_question += "\n\n" + "\n".join(
                                f"- {o}" for o in options
                            )
                        return state

                    state.tools_used.append("search_scientific_papers")
                    state.papers = structured.get("papers") or []
                    state.evidence.extend(evs)

            except Exception as exc:
                print(f"Tool {tool} failed: {exc}")
                state.limitations.append(f"{tool} search encountered an error.")

        if arxiv_weak and _should_continue_after_weak_arxiv(state):
            self._run_web_fallback(state, state.user_query)

        if (
            state.depth == "deep"
            and state.intent
            and "arxiv" in state.intent.tools_optional
            and "search_scientific_papers" not in state.tools_used
        ):
            try:
                _, structured, evs = self.run_arxiv(
                    query,
                    user_query=state.user_query,
                    depth=state.depth,
                    recent_only=recent,
                )
                state.paper_search = structured
                if not _arxiv_weak_or_clarification(structured):
                    state.tools_used.append("search_scientific_papers")
                    state.papers = structured.get("papers") or []
                    state.evidence.extend(evs)
            except Exception as exc:
                print(f"Optional arXiv (deep) failed: {exc}")

        return state


def calculator(expression: str) -> str:
    try:
        expression = expression.replace("pi", str(math.pi))
        expression = expression.replace("e", str(math.e))
        expression = expression.replace("sqrt", "math.sqrt")
        expression = expression.replace("sin", "math.sin")
        expression = expression.replace("cos", "math.cos")
        expression = expression.replace("tan", "math.tan")
        expression = expression.replace("log", "math.log10")
        expression = expression.replace("ln", "math.log")
        expression = expression.replace("exp", "math.exp")
        expression = expression.replace("abs", "abs")
        safe_dict = {
            "math": math,
            "__builtins__": {},
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "pow": pow,
        }
        return str(eval(expression, safe_dict))
    except Exception as e:
        return f"Calculation error: {str(e)}"


def _extract_calc_expression(query: str) -> str | None:
    patterns = [
        r"(?:calculate|compute|eval(?:uate)?)\s+(.+)",
        r"(?:what is|how much is)\s+([\d\s+\-*/().^]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    calc_pattern = r"[\d\+\-\*/\(\)\.\^\s]+"
    matches = re.findall(calc_pattern, query)
    if matches and any(c in query for c in "+-*/="):
        return max(matches, key=len).strip()
    return None
