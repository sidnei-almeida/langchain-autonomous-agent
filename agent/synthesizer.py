"""Answer synthesis from ranked evidence."""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from agent.evidence import format_evidence_for_llm
from agent.messages import build_system_prompt, format_messages_for_groq, dicts_to_langchain_messages
from agent.prompts import SYNTHESIS_INSTRUCTION
from agent.state import ResearchState


def synthesize_answer(llm, state: ResearchState) -> ResearchState:
    if state.intent and state.intent.needs_clarification:
        state.answer = state.intent.clarification_question
        state.confidence = 0.3
        state.limitations.append("Query needs clarification before research can proceed.")
        return state

    evidence_text = format_evidence_for_llm(state.evidence)
    plan_text = "\n".join(f"- {step}" for step in state.research_plan)

    synthesis_prompt = f"""User question: {state.user_query}

Research plan executed:
{plan_text}

{evidence_text}

{SYNTHESIS_INSTRUCTION}
"""

    messages = format_messages_for_groq(
        build_system_prompt(),
        state.conversation_history,
        synthesis_prompt,
    )
    context = dicts_to_langchain_messages(messages)

    try:
        response = llm.invoke(context)
        state.answer = (
            response.content if hasattr(response, "content") else str(response)
        )
        state.messages = context
    except Exception as exc:
        state.answer = _fallback_answer(state, str(exc))
        state.limitations.append("LLM synthesis unavailable; showing evidence summary.")
        state.messages = context

    return state


def _fallback_answer(state: ResearchState, error: str) -> str:
    if state.papers:
        lines = [state.paper_search.get("message", "Ranked arXiv results:"), ""]
        for i, p in enumerate(state.papers, 1):
            lines.append(f"{i}. {p['title']} ({p.get('year', '?')})")
            lines.append(f"   URL: {p['url']}")
            lines.append(f"   {p.get('whyItMatches', '')}")
        lines.append("\n(LLM synthesis unavailable.)")
        return "\n".join(lines)

    if state.evidence:
        lines = ["Evidence summary (LLM unavailable):", ""]
        for i, e in enumerate(state.evidence, 1):
            lines.append(f"{i}. [{e.source_type}] {e.title}")
            if e.url:
                lines.append(f"   {e.url}")
            lines.append(f"   {e.snippet[:300]}")
        return "\n".join(lines)

    return (
        "Unable to generate a full answer right now. "
        "Please retry with a clearer scientific question."
    )
