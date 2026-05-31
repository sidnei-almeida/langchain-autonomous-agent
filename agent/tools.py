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
from agent.state import EvidenceItem, ResearchState

RECENT_KEYWORDS = ("latest", "recent", "current", "new", "2024", "2025", "2026")


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

    def execute_tools(self, state: ResearchState) -> ResearchState:
        if not state.intent:
            return state

        query = state.intent.query_rewrite or state.user_query
        tools = state.intent.tools_required
        recent = any(kw in query.lower() for kw in RECENT_KEYWORDS)

        for tool in tools:
            try:
                if tool == "calculator":
                    text, ev = self.run_calculator(query)
                    state.tools_used.append("calculator")
                    if ev:
                        state.evidence.append(ev)

                elif tool == "wikipedia":
                    text, evs = self.run_wikipedia(query)
                    state.tools_used.append("wikipedia")
                    state.evidence.extend(evs)

                elif tool == "web_search":
                    text, evs = self.run_web(query)
                    state.tools_used.append("web_search")
                    state.evidence.extend(evs)

                elif tool == "arxiv":
                    text, structured, evs = self.run_arxiv(
                        query,
                        user_query=state.user_query,
                        depth=state.depth,
                        recent_only=recent,
                    )
                    state.tools_used.append("search_scientific_papers")
                    state.paper_search = structured
                    if structured.get("type") == "clarification":
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
                    state.papers = structured.get("papers") or []
                    state.evidence.extend(evs)

            except Exception as exc:
                print(f"Tool {tool} failed: {exc}")
                state.limitations.append(f"{tool} search encountered an error.")

        return state
