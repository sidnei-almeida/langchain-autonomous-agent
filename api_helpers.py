"""Build enriched API responses from agent results."""
from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage


def extract_answer(messages: list) -> str | None:
    return next(
        (m.content for m in reversed(messages) if isinstance(m, AIMessage)),
        None,
    )


def build_enriched_response(result: dict[str, Any], question: str | None = None) -> dict[str, Any]:
    messages = result.get("messages", [])
    answer = extract_answer(messages) or "No response generated."

    sources = result.get("sources") or []
    papers = result.get("papers") or []

    # Legacy structured block for older clients
    legacy_sources = [s.get("url") for s in sources if s.get("url")]
    authors: list[str] = []
    for p in papers:
        authors.extend(p.get("authors") or [])

    payload: dict[str, Any] = {
        "answer": answer,
        "tools_used": result.get("tools_used") or None,
        "intent": result.get("intent"),
        "research_plan": result.get("research_plan") or [],
        "sources": sources,
        "papers": papers,
        "confidence": result.get("confidence", 0.5),
        "limitations": result.get("limitations") or [],
        "follow_up_questions": result.get("follow_up_questions") or [],
        "structured": {
            "sources": legacy_sources[:20] or None,
            "authors": list(dict.fromkeys(authors))[:10] or None,
        },
    }

    if question is not None:
        payload["question"] = question

    return payload
