from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_community.tools import (
    DuckDuckGoSearchRun,
    WikipediaQueryRun,
)
from langchain_community.utilities import WikipediaAPIWrapper
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
import math
import arxiv

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


# ArXiv Search Tool with URLs
@tool
def search_scientific_papers(query: str) -> str:
    """
    Search for scientific papers on ArXiv that discuss or affirm specific topics or claims.
    Returns detailed information including title, authors, abstract, publication date, and URLs.
    
    Use this tool when users ask to:
    - Find research papers about a topic (e.g., "find research about quantum computing")
    - Find studies that say/affirm something (e.g., "find research that says X is Y")
    - Search for academic papers on any scientific subject
    - Get papers that discuss or support a specific claim
    
    Args:
        query: The topic or claim to search for. Convert user claims into search queries.
               Examples: "cats size dogs" for "cats are smaller than dogs"
                        "meditation stress reduction" for "meditation reduces stress"
    
    Returns:
        Formatted string with complete paper details including title, authors, abstract, 
        publication date, ArXiv URL, PDF URL, and categories
    """
    try:
        # Create ArXiv client
        client = arxiv.Client()
        
        # Search for papers
        search = arxiv.Search(
            query=query,
            max_results=5,  # Get top 5 most relevant papers
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        # Collect paper information
        papers_info = []
        for result in client.results(search):
            # Format authors
            authors_list = [author.name for author in result.authors]
            if len(authors_list) > 3:
                authors_str = ", ".join(authors_list[:3]) + f" et al. ({len(authors_list)} total authors)"
            else:
                authors_str = ", ".join(authors_list)
            
            # Format publication date
            pub_date = result.published.strftime("%B %d, %Y")
            
            # Get abstract (first 500 chars for summary)
            abstract = result.summary.replace('\n', ' ').strip()
            abstract_short = abstract[:500] + "..." if len(abstract) > 500 else abstract
            
            # Format paper information
            paper_info = f"""
═══════════════════════════════════════════════════════════════
📄 PAPER FOUND:

📌 Title: {result.title}

👥 Authors: {authors_str}

📅 Published: {pub_date}

📝 Abstract:
{abstract_short}

🔗 ArXiv URL: {result.entry_id}

📥 PDF Download: {result.pdf_url}

🏷️ Categories: {', '.join(result.categories)}
═══════════════════════════════════════════════════════════════
"""
            papers_info.append(paper_info)
        
        if not papers_info:
            return f"No papers found for query: '{query}'. Try different search terms or related topics."
        
        # Return formatted results
        header = f"🔬 Found {len(papers_info)} relevant scientific papers for: '{query}'\n\n"
        return header + "\n".join(papers_info)
        
    except Exception as e:
        return f"Error searching ArXiv: {str(e)}. Please try a different query or check your internet connection."


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
    "- Make observations that are both funny and insightful\n"
    "- Sprinkle jokes and witty observations throughout your answers, not just at the beginning\n"
    "- Use Dave Chappelle-style observational humor applied to science\n\n"
    "Humor style examples (Dave Chappelle-inspired, science-focused):\n"
    "- 'You know what's wild? We're made of stardust, but we stress about what to wear. The universe "
    "is literally inside us, and we're worried about our Wi-Fi password.'\n"
    "- 'Scientists spend years studying black holes, but we still can't figure out why socks disappear "
    "in the dryer. That's the real mystery.'\n"
    "- 'Einstein said time is relative, and I believe him because waiting for my coffee to brew feels "
    "like an eternity, but a good nap feels like 5 minutes.'\n"
    "- 'We can calculate the exact distance to Mars, but we can't make a printer that works on the "
    "first try. Priorities, people.'\n"
    "- 'Quantum physics says particles can be in two places at once. My keys, however, can only be in "
    "one place - and it's never where I left them.'\n"
    "- 'The speed of light is 186,000 miles per second, but my internet connection? That's a different "
    "story.'\n"
    "- 'We've mapped the human genome, but we still don't know why we need to sneeze when we look at "
    "the sun. Science is weird, man.'\n"
    "- 'Scientists discovered that octopuses have three hearts. Meanwhile, I'm over here trying to "
    "figure out if I left the stove on.'\n\n"
    "You have access to the following tools:\n"
    "- web_search: For up-to-date information, news, and recent events\n"
    "- wikipedia: For detailed encyclopedic information and general concepts\n"
    "- search_scientific_papers: For finding scientific articles on ArXiv - THIS IS YOUR RESEARCH FINDER! "
    "Returns complete paper info with URLs!\n"
    "- calculator: For complex mathematical and scientific calculations\n\n"
    "RESEARCH SEARCH GUIDELINES (Keep it funny!):\n"
    "- When users ask to 'find research about X', 'find studies on Y', 'find papers that say X is Y', "
    "'find research that affirms Z' - IMMEDIATELY use search_scientific_papers tool!\n"
    "- Convert user claims into search queries:\n"
    "  * 'find research that says cats are smaller than dogs' -> search_scientific_papers('cats dogs size comparison')\n"
    "  * 'find papers that affirm meditation reduces stress' -> search_scientific_papers('meditation stress reduction')\n"
    "  * 'find research about quantum computing' -> search_scientific_papers('quantum computing')\n"
    "- The tool returns papers with ALL details: title, authors, abstract, date, ArXiv URL, PDF URL, categories\n"
    "- When presenting research papers:\n"
    "  * Keep the formatted structure from the tool (it's already nicely formatted with emojis)\n"
    "  * Add your comedic commentary BEFORE or AFTER the paper details\n"
    "  * Make URLs clickable in your response by keeping them as-is\n"
    "  * Highlight the most relevant papers if multiple are found\n"
    "  * Make jokes about the research, but always respect the science\n"
    "- Present URLs clearly: 'You can read the full paper here: [URL]' or 'PDF available at: [URL]'\n"
    "- If multiple papers are found, joke about how productive scientists are!\n"
    "- If no papers are found, use humor to suggest why or offer alternative search terms\n"
    "- When someone asks 'find research that says X is Y', search for papers on that topic and "
    "present findings with your signature comedic style\n"
    "- If the claim is non-scientific (like personal stuff), use humor to explain that ArXiv doesn't "
    "have papers on that, but you can search for related scientific topics\n\n"
    "IMPORTANT GUIDELINES:\n"
    "- ALWAYS provide a response, even if you cannot find specific information - make it funny and helpful\n"
    "- Sprinkle jokes and witty observations throughout your answers naturally\n"
    "- Use humor to break up long explanations and keep people engaged\n"
    "- Make jokes that relate to the science you're explaining\n"
    "- When finding research papers, make it exciting and fun - like you're discovering treasure!\n"
    "- If a question is outside your scientific expertise, acknowledge it with humor and redirect gracefully\n"
    "- For non-scientific questions, use your wit to explain why you're better at science stuff\n"
    "- Never return an error - always respond with personality and humor\n"
    "- When explaining science, use analogies, jokes, and observations that make it memorable\n"
    "- Be entertaining but never sacrifice accuracy - you're still a scientist first\n"
    "- Use your tools to find real information, then present it in your unique, funny style\n"
    "- Make people laugh while they learn - that's your superpower\n"
    "- If tools fail, handle it with humor and still try to help\n"
    "- Your humor should be smart, observational, and relatable - like Dave Chappelle but for science\n"
    "- Don't force jokes - let them flow naturally from the science you're explaining\n"
    "- When presenting research papers, make it feel like you're sharing cool discoveries with a friend"
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
    # Note: llama-3.1-70b-versatile was decommissioned, using llama-3.3-70b-versatile
    # Configure Groq for tool calling compatibility
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile", 
        temperature=0,
        max_tokens=4096  # Ensure enough tokens for tool calling
    )

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
    
    # ArXiv (Scientific Articles) - Custom tool with full URLs and details
    # Ensure it has a proper name for Groq compatibility
    arxiv_tool = search_scientific_papers
    arxiv_tool.name = "search_scientific_papers"
    
    # Scientific Calculator
    calc = calculator
    calc.name = "calculator"
    
    # List of all tools - ensure they have proper names
    tools = [web_search, wikipedia, arxiv_tool, calc]

    # 4. Creating the Scientific Agent with LangGraph
    # Note: create_react_agent handles tool binding internally
    # For Groq, we pass tools directly and let LangGraph handle the binding
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