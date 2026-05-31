"""Shared state for the research pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ResearchDepth = Literal["quick", "standard", "deep"]
IntentType = Literal[
    "concept_explanation",
    "paper_search",
    "web_research",
    "technology_discovery",
    "tool_comparison",
    "calculation",
    "comparative_research",
    "mixed_research",
    "general_chat",
]


@dataclass
class IntentResult:
    intent: IntentType
    tools_required: list[str]
    tools_optional: list[str] = field(default_factory=list)
    needs_clarification: bool = False
    clarification_question: str = ""
    research_depth: ResearchDepth = "standard"
    query_rewrite: str = ""
    reason: str = ""


@dataclass
class EvidenceItem:
    title: str
    url: str
    source_type: Literal["arxiv", "wikipedia", "web", "calculation"]
    snippet: str
    relevance_score: float
    used_in_answer: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchState:
    user_query: str
    conversation_history: list[dict] = field(default_factory=list)
    depth: ResearchDepth = "standard"
    max_sources: int = 8

    intent: IntentResult | None = None
    research_plan: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)
    papers: list[dict[str, Any]] = field(default_factory=list)
    paper_search: dict[str, Any] | None = None

    answer: str = ""
    confidence: float = 0.5
    limitations: list[str] = field(default_factory=list)
    follow_up_questions: list[str] = field(default_factory=list)

    verification_passed: bool = True
    verification_notes: list[str] = field(default_factory=list)

    # LangChain compatibility
    messages: list[Any] = field(default_factory=list)

    def to_result_dict(self) -> dict[str, Any]:
        return {
            "messages": self.messages,
            "tools_used": self.tools_used,
            "intent": self.intent.intent if self.intent else None,
            "research_plan": self.research_plan,
            "sources": [
                {
                    "title": e.title,
                    "url": e.url,
                    "source_type": e.source_type,
                    "snippet": e.snippet,
                    "relevance_score": e.relevance_score,
                    "used_in_answer": e.used_in_answer,
                    "metadata": e.metadata,
                }
                for e in self.evidence
            ],
            "papers": self.papers,
            "confidence": self.confidence,
            "limitations": self.limitations,
            "follow_up_questions": self.follow_up_questions,
            "paper_search": self.paper_search,
        }
