"""Verify answers against available evidence."""
from __future__ import annotations

import re

from agent.prompts import VERIFIER_REVISION_INSTRUCTION
from agent.state import ResearchState

RECENCY_CLAIMS = ("latest", "recent", "current", "newest", "this year", "2024", "2025", "2026")
PAPER_CLAIMS = ("paper", "arxiv", "study", "studies", "research shows", "according to")


def verify_answer(state: ResearchState) -> ResearchState:
    issues: list[str] = []
    answer = state.answer.lower()
    evidence_urls = {e.url.lower() for e in state.evidence if e.url}
    evidence_titles = {e.title.lower() for e in state.evidence if e.title}

    # URLs in answer not in evidence
    mentioned_urls = set(re.findall(r"https?://[^\s\)\]]+", state.answer.lower()))
    for url in mentioned_urls:
        if url.rstrip(".,;") not in evidence_urls and "arxiv.org" in url:
            issues.append(f"Answer cites URL not in evidence: {url}")

    # Paper claims without arXiv evidence
    if any(p in answer for p in PAPER_CLAIMS):
        has_arxiv = any(e.source_type == "arxiv" for e in state.evidence)
        if not has_arxiv and state.intent and state.intent.intent in (
            "paper_search",
            "mixed_research",
        ):
            issues.append("Answer discusses papers but no arXiv evidence was retrieved.")

    # Recency claims without web/arxiv
    if any(r in answer for r in RECENCY_CLAIMS):
        has_fresh = any(
            e.source_type in ("web", "arxiv") for e in state.evidence
        ) or "search_scientific_papers" in state.tools_used or "web_search" in state.tools_used
        if not has_fresh:
            issues.append(
                "Answer uses recency language without web or arXiv search evidence."
            )

    state.verification_notes = issues
    state.verification_passed = len(issues) == 0

    if issues:
        state.limitations.extend(issues)
        state.answer = _add_caution(state.answer, issues)

    return state


def _add_caution(answer: str, issues: list[str]) -> str:
    if "limitations" in answer.lower():
        return answer
    caution = "\n\n**Limitations:** " + " ".join(issues[:2])
    return answer + caution


def revise_if_needed(llm, state: ResearchState) -> ResearchState:
    if state.verification_passed or not state.verification_notes:
        return state

    try:
        prompt = f"""Original answer:
{state.answer}

Verification issues:
{chr(10).join('- ' + i for i in state.verification_notes)}

{VERIFIER_REVISION_INSTRUCTION}
"""
        response = llm.invoke([
            SystemMessage(content="You are a careful scientific editor."),
            HumanMessage(content=prompt),
        ])
        revised = response.content if hasattr(response, "content") else str(response)
        if revised and len(revised) > 50:
            state.answer = revised
    except Exception as exc:
        print(f"Verifier revision failed: {exc}")

    return state
