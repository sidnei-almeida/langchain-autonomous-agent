"""Research pipeline orchestration."""
from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq

from agent.evidence import compute_confidence, mark_sources_used_in_answer, rank_evidence
from agent.messages import (
    build_system_prompt,
    dicts_to_langchain_messages,
    extract_conversation,
    format_messages_for_groq,
    is_heisenberg_name_response,
)
from agent.planner import build_research_plan
from agent.prompts import HEISENBERG_ACK_REPLY
from agent.router import classify_intent
from agent.state import ResearchState
from agent.synthesizer import synthesize_answer
from agent.tools import ToolRegistry
from agent.verifier import revise_if_needed, verify_answer


class GrayMatterResearchAgent:
    """Agentic research pipeline: classify → plan → tools → rank → synthesize → verify."""

    def __init__(self, llm, tools: ToolRegistry | None = None):
        self.llm = llm
        self.tools = tools or ToolRegistry()

    def research(
        self,
        query: str,
        *,
        depth: str = "standard",
        max_sources: int = 8,
        conversation_history: list | None = None,
    ) -> ResearchState:
        history, _ = extract_conversation(conversation_history or [])
        state = ResearchState(
            user_query=query,
            conversation_history=history,
            depth=depth if depth in ("quick", "standard", "deep") else "standard",
            max_sources=4 if depth == "quick" else max_sources,
        )

        if is_heisenberg_name_response(query):
            state.answer = HEISENBERG_ACK_REPLY
            state.messages = dicts_to_langchain_messages(
                format_messages_for_groq(build_system_prompt(), history, query)
            ) + [AIMessage(content=state.answer)]
            state.confidence = 1.0
            return state

        state.intent = classify_intent(self.llm, query, depth=state.depth)

        if state.intent.needs_clarification and state.intent.clarification_question:
            state.answer = state.intent.clarification_question
            state.confidence = 0.3
            state.research_plan = ["Clarify ambiguous query before searching"]
            state.messages = dicts_to_langchain_messages(
                format_messages_for_groq(build_system_prompt(), history, query)
            ) + [AIMessage(content=state.answer)]
            return state

        state.research_plan = build_research_plan(state.intent, query)
        state = self.tools.execute_tools(state)

        if state.intent.needs_clarification:
            state.answer = state.intent.clarification_question
            state.confidence = 0.3
            state.messages = dicts_to_langchain_messages(
                format_messages_for_groq(build_system_prompt(), history, query)
            ) + [AIMessage(content=state.answer)]
            return state

        state = rank_evidence(state)
        state = synthesize_answer(self.llm, state)
        state = verify_answer(state)
        state = revise_if_needed(self.llm, state)
        state.confidence = compute_confidence(state)
        mark_sources_used_in_answer(state, state.answer)

        state.follow_up_questions = _suggest_follow_ups(state)
        state.messages = (state.messages or []) + [AIMessage(content=state.answer)]
        return state

    def invoke(self, inputs: dict) -> dict:
        """LangChain-compatible invoke for /api/query and /api/chat."""
        raw = inputs.get("messages", [])
        history, query = extract_conversation(raw)

        if not query:
            return {
                "messages": dicts_to_langchain_messages(
                    format_messages_for_groq(build_system_prompt(), history, "")
                )
                + [AIMessage(content="Ask your question — science, research, or math.")],
                "tools_used": [],
            }

        depth = inputs.get("depth", "standard")
        max_sources = inputs.get("max_sources", 8)

        state = self.research(
            query,
            depth=depth,
            max_sources=max_sources,
            conversation_history=raw,
        )

        return state.to_result_dict()


def _suggest_follow_ups(state: ResearchState) -> list[str]:
    if state.intent and state.intent.needs_clarification:
        return []
    follow_ups: list[str] = []
    if state.papers:
        follow_ups.append("Would you like a deeper summary of any specific paper?")
    if state.intent and state.intent.intent == "concept_explanation":
        follow_ups.append("Would you like recent papers on this topic?")
    if state.intent and state.intent.intent in (
        "mixed_research",
        "web_research",
        "technology_discovery",
        "tool_comparison",
    ):
        follow_ups.append("Should I search arXiv for peer-reviewed sources?")
    return follow_ups[:3]


def create_scientific_agent() -> GrayMatterResearchAgent:
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Set it in .env or platform secrets (HF Spaces)."
        )

    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.2,
        max_tokens=1024,
        model_kwargs={"top_p": 0.9},
    )
    return GrayMatterResearchAgent(llm)
