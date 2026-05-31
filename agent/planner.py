"""Research planning step."""
from __future__ import annotations

from agent.state import IntentResult

PLAN_TEMPLATES: dict[str, list[str]] = {
    "concept_explanation": [
        "Identify the core concept",
        "Retrieve encyclopedic background",
        "Synthesize a clear explanation with sources",
    ],
    "paper_search": [
        "Identify the research topic",
        "Search arXiv for ranked papers",
        "Summarize findings with paper links",
    ],
    "web_research": [
        "Identify the information need",
        "Search the web for current context",
        "Synthesize answer with sources",
    ],
    "technology_discovery": [
        "Identify the technology or tooling question",
        "Search the web for current tools and practices",
        "Synthesize answer with web sources (not arXiv-first)",
    ],
    "tool_comparison": [
        "Identify tools or platforms to compare",
        "Search the web for comparisons and production guidance",
        "Structure a comparison with evidence from web sources",
    ],
    "calculation": [
        "Parse the mathematical expression",
        "Run deterministic calculation",
        "Return the numeric result",
    ],
    "comparative_research": [
        "Identify topics to compare",
        "Gather web and literature context",
        "Structure a comparison with evidence",
    ],
    "mixed_research": [
        "Identify the scientific topic",
        "Search arXiv for recent papers",
        "Search web for current context",
        "Synthesize answer with sources",
    ],
    "general_chat": [
        "Understand the user question",
        "Answer from general knowledge or prior context",
    ],
}


def build_research_plan(intent: IntentResult, query: str) -> list[str]:
    """Concise operational trace (not chain-of-thought)."""
    base = list(PLAN_TEMPLATES.get(intent.intent, PLAN_TEMPLATES["general_chat"]))

    if "arxiv" in intent.tools_required and "Search arXiv" not in " ".join(base):
        base.insert(-1, "Search arXiv for relevant papers")
    if "web_search" in intent.tools_required and "web" not in " ".join(base).lower():
        base.insert(-1, "Search web for current context")
    if "wikipedia" in intent.tools_required and "Wikipedia" not in " ".join(base):
        base.insert(1, "Search Wikipedia for background")

    return base[:5]
