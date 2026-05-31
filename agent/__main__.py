"""CLI entry point: python -m agent"""
from __future__ import annotations

import sys

from langchain_core.messages import AIMessage, HumanMessage

from agent import create_scientific_agent, prepare_messages


def main() -> None:
    ag = create_scientific_agent()

    print("\n" + "=" * 60)
    print("  GRAY MATTER RESEARCH AGENT")
    print("=" * 60)
    print("\nTools: Web · Wikipedia · arXiv (ranked) · Calculator")
    print("-" * 60)

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Your question: ").strip()
        if not question:
            question = "What are the latest advances in agentic RAG?"

    print(f"\nQuestion: {question}\nProcessing...\n")

    try:
        result = ag.invoke({
            "messages": prepare_messages([HumanMessage(content=question)]),
            "depth": "standard",
        })
        answer = next(
            (m.content for m in reversed(result.get("messages", [])) if isinstance(m, AIMessage)),
            None,
        )
        print("=" * 60)
        print("ANSWER")
        print("=" * 60)
        print(answer or "No response.")
        if result.get("research_plan"):
            print("\nPlan:", result["research_plan"])
        if result.get("tools_used"):
            print("Tools:", result["tools_used"])
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
