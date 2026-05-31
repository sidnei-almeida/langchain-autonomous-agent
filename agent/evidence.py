"""Evidence ranking and normalization."""
from __future__ import annotations

from agent.state import EvidenceItem, ResearchState


def rank_evidence(state: ResearchState) -> ResearchState:
    """Sort evidence by relevance and cap to max_sources."""
    state.evidence.sort(key=lambda e: e.relevance_score, reverse=True)

    seen_urls: set[str] = set()
    deduped: list[EvidenceItem] = []
    for item in state.evidence:
        key = item.url or item.title
        if key in seen_urls:
            continue
        seen_urls.add(key)
        deduped.append(item)

    state.evidence = deduped[: state.max_sources]
    return state


def format_evidence_for_llm(evidence: list[EvidenceItem]) -> str:
    if not evidence:
        return "No external evidence retrieved."

    lines = ["Evidence items (cite ONLY these sources):"]
    for i, item in enumerate(evidence, 1):
        lines.append(f"\n[{i}] type={item.source_type} | score={item.relevance_score:.2f}")
        lines.append(f"Title: {item.title}")
        if item.url:
            lines.append(f"URL: {item.url}")
        lines.append(f"Snippet: {item.snippet[:500]}")
        if item.metadata.get("whyItMatches"):
            lines.append(f"Why it matches: {item.metadata['whyItMatches']}")
    return "\n".join(lines)


def compute_confidence(state: ResearchState) -> float:
    if state.intent and state.intent.needs_clarification:
        return 0.3
    tooling_intents = (
        "technology_discovery",
        "tool_comparison",
        "web_research",
        "concept_explanation",
    )
    if not state.evidence:
        if state.intent and state.intent.intent in tooling_intents:
            return 0.4
        return 0.45 if state.intent and state.intent.intent == "general_chat" else 0.35
    avg = sum(e.relevance_score for e in state.evidence) / len(state.evidence)
    tool_bonus = min(0.2, len(state.tools_used) * 0.05)
    score = min(0.95, avg * 0.7 + tool_bonus + 0.15)
    if (
        state.intent
        and state.intent.intent in tooling_intents
        and any(e.source_type == "web" for e in state.evidence)
    ):
        score = max(score, 0.55)
    return round(score, 2)


def mark_sources_used_in_answer(state: ResearchState, answer: str) -> None:
    lower = answer.lower()
    for item in state.evidence:
        if item.url and item.url.lower() in lower:
            item.used_in_answer = True
        elif item.title and len(item.title) > 10 and item.title.lower()[:40] in lower:
            item.used_in_answer = True
