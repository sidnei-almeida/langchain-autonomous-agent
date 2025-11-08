from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_community.tools import (
    DuckDuckGoSearchRun,
    WikipediaQueryRun,
    ArxivQueryRun,
)
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
import math

# Scientific Calculator Tool
@tool
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


# System message for the scientific agent
SYSTEM_MESSAGE = (
    "You are a brilliant scientist and researcher with a sharp sense of humor and a talent for "
    "making complex topics entertaining. Think of yourself as a scientific comedian - you're like "
    "Dave Chappelle meets Neil deGrasse Tyson. You're smart, funny, observant, and you don't take "
    "yourself too seriously, but you always deliver accurate, evidence-based information.\n\n"
    "Your personality:\n"
    "- Witty and observant with clever observations about science and life\n"
    "- Use humor to make complex topics more accessible and engaging\n"
    "- Make smart jokes and analogies that help people understand science better\n"
    "- Be confident and charismatic in your explanations\n"
    "- Add personality and flair to your responses while staying accurate\n"
    "- Use casual, conversational language mixed with scientific precision\n"
    "- Make observations that are both funny and insightful\n\n"
    "You have access to the following tools:\n"
    "- web_search: For up-to-date information, news, and recent events\n"
    "- wikipedia: For detailed encyclopedic information and general concepts\n"
    "- arxiv: For scientific articles, academic papers, and scientific literature\n"
    "- calculator: For complex mathematical and scientific calculations\n\n"
    "IMPORTANT GUIDELINES:\n"
    "- ALWAYS provide a response, even if you cannot find specific information - make it funny and helpful\n"
    "- If a question is outside your scientific expertise, acknowledge it with humor and redirect gracefully\n"
    "- For non-scientific questions, use your wit to explain why you're better at science stuff\n"
    "- Never return an error - always respond with personality and humor\n"
    "- When explaining science, use analogies, jokes, and observations that make it memorable\n"
    "- Be entertaining but never sacrifice accuracy - you're still a scientist first\n"
    "- Use your tools to find real information, then present it in your unique, funny style\n"
    "- Make people laugh while they learn - that's your superpower\n"
    "- If tools fail, handle it with humor and still try to help\n"
    "- Your humor should be smart, not mean-spirited - you're here to educate and entertain"
)

def prepare_messages(messages):
    """Prepare messages with system message if not already present."""
    # Check if first message is already a SystemMessage
    if messages and isinstance(messages[0], SystemMessage):
        return messages
    
    # Add system message at the beginning
    return [SystemMessage(content=SYSTEM_MESSAGE)] + messages

def create_scientific_agent():
    """Creates and returns a configured scientific agent."""
    # 1. Initial Configuration
    # Load Groq API key from .env file (if exists) or environment variables
    # This works for both local development (.env) and Hugging Face Spaces (secrets)
    load_dotenv()
    
    # Verify API key is available
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Please set it in your .env file or "
            "as an environment variable (for Hugging Face Spaces, use secrets)."
        )

    # 2. The Brain (LLM)
    # We use Llama 3 70B because it's excellent at following complex instructions
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)

    # 3. Scientific Tools Configuration
    # Web Search (DuckDuckGo)
    web_search = DuckDuckGoSearchRun(
        name="web_search",
        description="Searches for up-to-date information on the internet using DuckDuckGo. Use for news, recent events, and general information."
    )
    
    # Wikipedia (Encyclopedia)
    wikipedia_api = WikipediaAPIWrapper()
    wikipedia = WikipediaQueryRun(
        api_wrapper=wikipedia_api,
        name="wikipedia",
        description="Searches for detailed and encyclopedic information on Wikipedia. Ideal for concepts, biographies, historical events, and in-depth explanations."
    )
    
    # ArXiv (Scientific Articles)
    arxiv_api = ArxivAPIWrapper()
    arxiv = ArxivQueryRun(
        api_wrapper=arxiv_api,
        name="arxiv",
        description="Searches and retrieves scientific articles from ArXiv. Use to find academic papers, recent research, and scientific literature on any topic."
    )
    
    # Scientific Calculator
    calc = calculator
    
    # List of all tools
    tools = [web_search, wikipedia, arxiv, calc]

    # 4. Creating the Scientific Agent with LangGraph
    # Note: create_react_agent doesn't accept state_modifier parameter
    # We'll add the system message to each conversation instead
    agent = create_react_agent(llm, tools)
    
    return agent


def main() -> None:
    """Main function for CLI usage."""
    agent = create_scientific_agent()
    
    print("🔬 Configuring scientific tools...")
    print("✅ 4 tools configured:")
    print("   🌐 Web Search (DuckDuckGo)")
    print("   📚 Wikipedia")
    print("   🔬 ArXiv (Scientific Articles)")
    print("   🧮 Scientific Calculator")

    # 5. Interactive Interface
    print("\n" + "="*60)
    print("🔬 AUTONOMOUS SCIENTIFIC AGENT")
    print("="*60)
    print("\nAvailable tools:")
    print("  🌐 Web Search (DuckDuckGo)")
    print("  📚 Wikipedia")
    print("  🔬 ArXiv (Scientific Articles)")
    print("  🧮 Scientific Calculator")
    print("\n" + "-"*60)
    
    # Interactive mode or single question
    import sys
    if len(sys.argv) > 1:
        # Command line argument mode
        question = " ".join(sys.argv[1:])
    else:
        # Interactive mode
        question = input("\n💬 Ask your scientific question: ").strip()
        if not question:
            question = "What are the latest advances in artificial intelligence according to ArXiv?"
    
    print(f"\n🔍 Question: {question}\n")
    print("🤔 Processing...\n")

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
        print("📊 FINAL ANSWER")
        print("="*60)
        if final_answer:
            print(final_answer)
        else:
            print("Could not get an answer from the agent.")
            print("\nReceived messages:")
            for msg in messages[-3:]:  # Show last 3 messages
                print(f"  - {type(msg).__name__}: {str(msg.content)[:100]}...")
    except Exception as e:
        print(f"\n❌ Error processing: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()