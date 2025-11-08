from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_community.tools import (
    DuckDuckGoSearchRun,
    WikipediaQueryRun,
    ArxivQueryRun,
)
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
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

    # 4. Creating the Scientific Agent with the new API (LangChain >= 1.0)
    agent = create_agent(
        llm,
        tools,
        system_prompt=(
            "You are a professional scientist and experienced researcher with access to multiple "
            "scientific research tools. Your mission is to provide accurate, well-founded, "
            "and evidence-based answers.\n\n"
            "You have access to the following tools:\n"
            "- Web Search: For up-to-date information, news, and recent events\n"
            "- Wikipedia: For detailed encyclopedic information and general concepts\n"
            "- ArXiv: For scientific articles, academic papers, and scientific literature\n"
            "- Calculator: For complex mathematical and scientific calculations\n\n"
            "Whenever possible, use multiple sources to validate information. "
            "Prioritize scientific articles from ArXiv for technical and scientific questions. "
            "Use the calculator for any necessary calculations. "
            "Be precise, cite your sources when relevant, and explain your reasoning."
        ),
    )
    
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
        result = agent.invoke({"messages": [{"role": "user", "content": question}]})
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