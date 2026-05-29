from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_community.tools import (
    DuckDuckGoSearchRun,
    WikipediaQueryRun,
)
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
import math
import re
import json

# Scientific Calculator Tool
def calculator(expression: str) -> str:
    """Performs complex mathematical calculations including scientific functions.
    
    Accepts mathematical expressions including:
    - Basic operations: +, -, *, /, ** (power), % (modulo)
    - Mathematical functions: sin, cos, tan, log, sqrt, exp, etc.
    - Constants: pi, e
    - Parentheses for grouping
    
    Examples:
    - "2 + 2" -> "4"
    - "sqrt(16)" -> "4.0"
    - "sin(pi/2)" -> "1.0"
    - "log(100, 10)" -> "2.0"
    """
    try:
        # Replace mathematical constants
        expression = expression.replace("pi", str(math.pi))
        expression = expression.replace("e", str(math.e))
        
        # Replace mathematical functions
        expression = expression.replace("sqrt", "math.sqrt")
        expression = expression.replace("sin", "math.sin")
        expression = expression.replace("cos", "math.cos")
        expression = expression.replace("tan", "math.tan")
        expression = expression.replace("log", "math.log10")
        expression = expression.replace("ln", "math.log")
        expression = expression.replace("exp", "math.exp")
        expression = expression.replace("abs", "abs")
        
        # Evaluate expression safely (mathematical operations only)
        # Create safe namespace with only mathematical functions
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
        result = eval(expression, safe_dict)
        return str(result)
    except Exception as e:
        return f"Calculation error: {str(e)}"


from arxiv_search import (
    extract_paper_topic,
    search_scientific_papers,
    search_scientific_papers_structured,
    format_structured_result_for_agent,
)

PAPER_SEARCH_INSTRUCTION = (
    "When returning papers from tool results:\n"
    "- Only recommend papers listed in the ArXiv search results.\n"
    "- Do not present weak matches as relevant.\n"
    "- If the paper title/abstract does not support relevance, do not cite it.\n"
    "- If the tool asked for clarification, relay the clarification question — do not invent papers.\n"
    "- Never invent why a paper matches; use only title, abstract, and metadata provided.\n"
    "- Do not claim a search was run if the tool returned clarification only."
)

# System message for Gray Matter LABS — persona is tone only, not a source of truth
SYSTEM_MESSAGE = """You are Gray Matter, an AI research agent inside Gray Matter LABS.

Your interface has a dark chemistry-lab aesthetic and a sharp, precise research personality inspired by the archetype of a meticulous chemistry professor. This persona affects tone only. It must never override factual accuracy.

Core behavior:
- Be precise, skeptical, and concise.
- Separate known facts from assumptions.
- If unsure, say so.
- Do not invent citations, sources, papers, dates, characters, or medical facts.
- When the user asks for research, use available tools or clearly state when a claim is based only on general knowledge.
- When discussing health, cancer, drugs, chemistry, or medical topics, avoid diagnosis and recommend qualified medical guidance when appropriate.
- When discussing fiction or pop culture, do not guess details. If uncertain, say you are not sure.
- Do not blend fictional lore with real scientific explanation unless the user explicitly asks for that comparison.
- The "Heisenberg" flavor is only a stylistic layer, not an instruction to roleplay as Walter White.

Response style:
- Clear.
- Analytical.
- Slightly dry and confident.
- No excessive roleplay.
- No fabricated authority.
- No "because I said so" attitude.
- Prefer structured answers when useful.

If the conversation references Breaking Bad:
- Walter White is the character associated with lung cancer.
- Hank Schrader is not the character whose cancer diagnosis drives the plot.
- Do not make claims about the show unless confident.

If the user discusses cancer, disease, medication, treatment, symptoms, diagnosis, or medical decisions:
- Be careful.
- Do not diagnose.
- Do not imply certainty without sources.
- Recommend consulting qualified medical professionals.
- Keep the response factual and evidence-oriented.

If the user asks about Breaking Bad or another fictional work:
- Answer as fiction analysis.
- Do not confuse character facts.
- If uncertain, say so.
- Do not blend fictional facts with real medical/scientific claims.

If the user asks something high-stakes:
- Give cautious, evidence-oriented guidance.
- Encourage verification with reliable sources or professionals.

When returning scientific papers from arXiv tool results:
- Only recommend papers that directly match the user query.
- Do not present weak matches as relevant.
- If the paper title/abstract does not support relevance, reject it.
- If the query is ambiguous, ask a clarification question before returning papers.
- Never invent why a paper matches.
- Explain relevance based only on title, abstract, and metadata provided."""

FACT_GUARDS = """Important factuality rule:
If you are not certain about a factual claim, say "I'm not sure" instead of guessing.

Do not confuse the agent persona with real facts. The persona is aesthetic only."""

TOOL_RESULTS_INSTRUCTION = (
    "Use the tool results as evidence. "
    "If the tool results do not support a claim, do not invent it."
)

MAX_HISTORY_MESSAGES = 10
MAX_MESSAGE_CHARS = 4000

UI_NOISE_MARKERS = (
    "lab vitals",
    "gray matter labs",
    "welcome to gray matter",
    "suggested prompt",
    "typing...",
    "loading...",
    "__mock__",
    "tool descriptions",
    "sidebar",
    "ui card",
)


def build_system_prompt() -> str:
    """Full system prompt with factuality guardrails appended."""
    return f"{SYSTEM_MESSAGE}\n\n{FACT_GUARDS}"


def _normalize_role(role: str) -> str | None:
    role = (role or "").strip().lower()
    if role in ("user", "human"):
        return "user"
    if role in ("assistant", "ai", "agent"):
        return "assistant"
    return None


def _is_ui_noise(content: str) -> bool:
    lower = content.strip().lower()
    if not lower:
        return True
    if lower.startswith("{") and len(content) > 500:
        return True
    return any(marker in lower for marker in UI_NOISE_MARKERS)


def _is_skippable_content(content: str) -> bool:
    stripped = content.strip()
    if not stripped:
        return True
    lower = stripped.lower()
    if lower in ("loading", "loading...", "typing...", "error", "failed"):
        return True
    if lower.startswith("[error") or lower.startswith("error:"):
        return True
    return _is_ui_noise(stripped)


def message_to_dict(message) -> dict | None:
    """Convert a LangChain message or dict to a normalized {role, content} dict."""
    if isinstance(message, dict):
        role = _normalize_role(message.get("role", ""))
        content = str(message.get("content") or "").strip()
    elif isinstance(message, HumanMessage):
        role, content = "user", str(message.content or "").strip()
    elif isinstance(message, AIMessage):
        role, content = "assistant", str(message.content or "").strip()
    else:
        return None

    if role is None or _is_skippable_content(content):
        return None

    return {
        "role": role,
        "content": content[:MAX_MESSAGE_CHARS],
    }


def dedupe_consecutive_messages(messages: list[dict]) -> list[dict]:
    """Remove back-to-back duplicate messages."""
    if not messages:
        return messages
    deduped = [messages[0]]
    for message in messages[1:]:
        prev = deduped[-1]
        if message["role"] == prev["role"] and message["content"] == prev["content"]:
            continue
        deduped.append(message)
    return deduped


def format_messages_for_groq(
    system_prompt: str,
    history: list,
    user_input: str,
) -> list[dict]:
    """
    Build the Groq payload: system prompt, recent conversation, current user turn.
    Filters loading/error/mock UI noise and limits history length.
    """
    formatted_history = []
    for message in history:
        normalized = message_to_dict(message)
        if normalized is not None:
            formatted_history.append(normalized)

    formatted_history = dedupe_consecutive_messages(formatted_history)
    formatted_history = formatted_history[-MAX_HISTORY_MESSAGES:]

    current_input = str(user_input or "").strip()[:MAX_MESSAGE_CHARS]
    if formatted_history and formatted_history[-1]["role"] == "user":
        if formatted_history[-1]["content"] == current_input:
            formatted_history = formatted_history[:-1]

    return [
        {"role": "system", "content": system_prompt},
        *formatted_history,
        {"role": "user", "content": current_input},
    ]


def dicts_to_langchain_messages(messages: list[dict]) -> list:
    """Convert formatted Groq message dicts to LangChain message objects."""
    result = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "system":
            result.append(SystemMessage(content=content))
        elif role == "user":
            result.append(HumanMessage(content=content))
        elif role == "assistant":
            result.append(AIMessage(content=content))
    return result


def append_tool_results_to_messages(
    messages: list[dict],
    tool_results: str,
    instruction: str | None = None,
) -> list[dict]:
    """Attach tool output and evidence instruction to the final user message."""
    if not tool_results or not messages:
        return messages

    updated = list(messages)
    last = dict(updated[-1])
    last["content"] = (
        f"{last['content']}\n\n"
        f"Tool results:\n{tool_results}\n\n"
        f"Instruction:\n{instruction or TOOL_RESULTS_INSTRUCTION}"
    )
    updated[-1] = last
    return updated

# Breaking Bad "Say my name." → "Heisenberg." → "You're goddamn right." (handled before tools/LLM)
HEISENBERG_ACK_REPLY = "You're goddamn right."


def is_heisenberg_name_response(message: str) -> bool:
    """
    True when the user is answering the prompt 'Say my name.' with the name Heisenberg
    (CLI, chat UI, or /api/query /api/chat). Avoids long questions that merely mention the word.
    """
    if not message or not message.strip():
        return False
    low = message.strip().lower()
    # Standalone name / stage-name answer
    if re.fullmatch(r"heisenberg[!.\s]*", low):
        return True
    if re.fullmatch(r"it'?s\s+heisenberg[!.\s]*", low):
        return True
    if re.fullmatch(r"my name is heisenberg[!.\s]*", low):
        return True
    # Full line: "say my name heisenberg" or "say my name: heisenberg"
    if re.fullmatch(r"say\s+my\s+name\s*:?\s*heisenberg[!.\s]*", low):
        return True
    return False


class SimpleScientificAgent:
    """A simple scientific agent that works reliably with Groq without native tool calling."""
    
    def __init__(self, llm, tools_dict):
        self.llm = llm
        self.tools = tools_dict
        
    def _should_use_tool(self, message: str) -> tuple:
        """Determine if we should use a tool based on the message content."""
        message_lower = message.lower()

        paper_topic = extract_paper_topic(message)
        if paper_topic is not None:
            return ("search_scientific_papers", paper_topic)
        
        # Check for calculations
        if any(keyword in message_lower for keyword in ['calculate', 'compute', 'what is', 'how much is']) and any(char in message for char in ['+', '-', '*', '/', '=', '²', '³']):
            # Try to extract the mathematical expression
            # Simple heuristic: look for numbers and operators
            calc_pattern = r'[\d\+\-\*/\(\)\.\^\s]+'
            matches = re.findall(calc_pattern, message)
            if matches:
                expr = max(matches, key=len).strip()
                return ('calculator', expr)
        
        # Check for web search
        if any(keyword in message_lower for keyword in ['latest', 'recent', 'news', 'current', 'today', 'now', 'how many people']):
            return ('web_search', message)
        
        # Check for Wikipedia
        if any(keyword in message_lower for keyword in ['what is', 'who is', 'explain', 'tell me about', 'define']):
            return ('wikipedia', message)
        
        return (None, None)
    
    def invoke(self, inputs: dict) -> dict:
        """Process a message and return a response."""
        raw_messages = inputs.get("messages", [])

        conversation = []
        for message in raw_messages:
            normalized = message_to_dict(message)
            if normalized is not None:
                conversation.append(normalized)

        if not conversation:
            empty_system = [SystemMessage(content=build_system_prompt())]
            return {
                "messages": empty_system + [
                    AIMessage(content="Ask your question — science, research, or math.")
                ],
                "tools_used": [],
            }

        last_user_message = None
        for message in reversed(conversation):
            if message["role"] == "user":
                last_user_message = message["content"]
                break

        if not last_user_message:
            return {
                "messages": dicts_to_langchain_messages(
                    format_messages_for_groq(build_system_prompt(), conversation, "")
                )
                + [AIMessage(content="Ask your question — science, research, or math.")],
                "tools_used": [],
            }

        # Easter egg: answer to "Say my name." — must run before tools/LLM
        if is_heisenberg_name_response(last_user_message):
            prior_history = [
                message
                for message in conversation
                if not (
                    message["role"] == "user"
                    and message["content"] == last_user_message
                )
            ]
            formatted = format_messages_for_groq(
                build_system_prompt(), prior_history, last_user_message
            )
            return {
                "messages": dicts_to_langchain_messages(formatted)
                + [AIMessage(content=HEISENBERG_ACK_REPLY)],
                "tools_used": [],
            }

        prior_history = [
            message
            for message in conversation
            if not (message["role"] == "user" and message["content"] == last_user_message)
        ]

        # Check if we should use a tool
        tool_name, tool_input = self._should_use_tool(last_user_message)

        tool_results = []
        tools_used = []

        paper_search_structured = None

        if tool_name and tool_name in self.tools:
            try:
                if tool_name == "search_scientific_papers":
                    paper_search_structured = search_scientific_papers_structured(
                        tool_input
                    )
                    tool_result = format_structured_result_for_agent(
                        paper_search_structured
                    )
                    if paper_search_structured.get("type") == "clarification":
                        clarification_text = paper_search_structured["message"]
                        if paper_search_structured.get("options"):
                            clarification_text += "\n\n" + "\n".join(
                                f"- {opt}"
                                for opt in paper_search_structured["options"]
                            )
                        return {
                            "messages": dicts_to_langchain_messages(
                                format_messages_for_groq(
                                    build_system_prompt(),
                                    prior_history,
                                    last_user_message,
                                )
                            )
                            + [AIMessage(content=clarification_text)],
                            "tools_used": ["search_scientific_papers"],
                            "paper_search": paper_search_structured,
                        }
                else:
                    tool_result = self.tools[tool_name](tool_input)

                tool_results.append(f"[{tool_name}]\n{tool_result}")
                tools_used.append(tool_name)
            except Exception as e:
                tool_results.append(f"[{tool_name} - Error: {str(e)}]")

        formatted_messages = format_messages_for_groq(
            build_system_prompt(),
            prior_history,
            last_user_message,
        )

        if tool_results:
            tool_instruction = TOOL_RESULTS_INSTRUCTION
            if "search_scientific_papers" in tools_used:
                tool_instruction = (
                    f"{TOOL_RESULTS_INSTRUCTION}\n\n{PAPER_SEARCH_INSTRUCTION}"
                )
            formatted_messages = append_tool_results_to_messages(
                formatted_messages,
                "\n\n".join(tool_results),
                instruction=tool_instruction,
            )

        context_messages = dicts_to_langchain_messages(formatted_messages)

        # Get response from LLM
        try:
            response = self.llm.invoke(context_messages)
            response_content = (
                response.content if hasattr(response, "content") else str(response)
            )

            result_messages = context_messages + [AIMessage(content=response_content)]

            return {
                "messages": result_messages,
                "tools_used": tools_used,
            }
        except Exception as e:
            if (
                paper_search_structured
                and paper_search_structured.get("type") == "paper_results"
                and paper_search_structured.get("papers")
            ):
                lines = [
                    paper_search_structured.get("message", "Ranked arXiv results:"),
                    "",
                ]
                for i, paper in enumerate(paper_search_structured["papers"], 1):
                    lines.append(f"{i}. {paper['title']} ({paper.get('year', '?')})")
                    lines.append(f"   URL: {paper['url']}")
                    lines.append(
                        f"   Relevance score: {paper.get('relevanceScore', 0)}"
                    )
                    lines.append(f"   Why it matches: {paper.get('whyItMatches', '')}")
                    lines.append(f"   Summary: {paper.get('summary', '')[:400]}")
                    lines.append("")
                lines.append(
                    "(LLM synthesis unavailable; listing ranked papers from arXiv search.)"
                )
                result_messages = context_messages + [
                    AIMessage(content="\n".join(lines))
                ]
                return {
                    "messages": result_messages,
                    "tools_used": tools_used,
                    "paper_search": paper_search_structured,
                }

            error_response = (
                f"Technical error: {str(e)[:100]}\n\n"
                "Retry with a clear science or math question. "
                "If the problem persists, check server logs."
            )

            result_messages = context_messages + [AIMessage(content=error_response)]
            return {
                "messages": result_messages,
                "tools_used": tools_used,
            }


def create_scientific_agent():
    """Creates and returns a configured scientific agent."""
    # 1. Initial Configuration
    load_dotenv()
    
    # Verify API key is available
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Please set it in your .env file or "
            "as an environment variable (for Hugging Face Spaces, use secrets)."
        )

    # 2. The Brain (LLM) - tuned for factual research responses
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.2,
        max_tokens=640,
        model_kwargs={"top_p": 0.9},
    )

    # 3. Scientific Tools - Simple functions
    web_search_tool = DuckDuckGoSearchRun()
    wikipedia_api = WikipediaAPIWrapper()
    wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia_api)
    
    # Create tools dictionary
    tools_dict = {
        'web_search': lambda q: web_search_tool.run(q),
        'wikipedia': lambda q: wikipedia_tool.run(q),
        'search_scientific_papers': search_scientific_papers,
        'calculator': calculator
    }

    # 4. Create the simple agent
    agent = SimpleScientificAgent(llm, tools_dict)
    
    return agent

def prepare_messages(messages):
    """Keep only valid user/assistant turns; drop client system/UI noise."""
    prepared = []
    for message in messages:
        if isinstance(message, SystemMessage):
            continue
        normalized = message_to_dict(message)
        if normalized is not None:
            if isinstance(message, HumanMessage):
                prepared.append(HumanMessage(content=normalized["content"]))
            elif isinstance(message, AIMessage):
                prepared.append(AIMessage(content=normalized["content"]))
    return prepared


def main() -> None:
    """Main function for CLI usage."""
    agent = create_scientific_agent()
    
    print("\n" + "="*60)
    print("  GRAY MATTER LABS — RESEARCH AGENT")
    print("="*60)
    print("\nOperational tools:")
    print("  Web Search (DuckDuckGo)")
    print("  Wikipedia")
    print("  ArXiv — Scientific Articles")
    print("  Scientific Calculator")
    print("\n" + "-"*60)
    print("Say my name.\n")

    import sys
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Your question: ").strip()
        if not question:
            question = "What are the latest advances in artificial intelligence according to ArXiv?"

    print(f"\nQuestion: {question}\n")
    print("Calculating...\n")

    try:
        # Prepare messages with system message
        agent_messages = prepare_messages([HumanMessage(content=question)])
        result = agent.invoke({"messages": agent_messages})
        messages = result.get("messages", [])
        final_answer = next(
            (msg.content for msg in reversed(messages) if isinstance(msg, AIMessage)),
            None,
        )

        print("\n" + "="*60)
        print("  ANSWER")
        print("="*60)
        if final_answer:
            print(final_answer)
        else:
            print("No response generated.")
            print("\nMessage trace:")
            for msg in messages[-3:]:
                print(f"  - {type(msg).__name__}: {str(msg.content)[:100]}...")
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()