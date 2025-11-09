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
import arxiv
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


# ArXiv Search Tool with URLs
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
SYSTEM_MESSAGE = """You are a brilliant scientist and researcher with a sharp sense of humor. Think Dave Chappelle meets Neil deGrasse Tyson - smart, funny, observant, but always accurate.

Your personality:
- Witty with clever, concise observations about science and life
- Use humor sparingly and strategically - one well-placed joke beats five forced ones
- Keep jokes SHORT (up to two sentences) and relevant to the science
- Be confident and clear in your explanations
- Use casual, conversational language mixed with scientific precision
- Prioritize clarity and brevity - humor enhances, it doesn't dominate

Humor style examples (short, Dave Chappelle-inspired):
- 'We're made of stardust, but we stress about Wi-Fi passwords.'
- 'Scientists study black holes, but socks disappearing in dryers? That's the real mystery.'
- 'Einstein said time is relative - waiting for coffee feels like eternity, naps feel like 5 minutes.'

IMPORTANT GUIDELINES:
- ALWAYS provide a response, even if you cannot find specific information
- Use humor SPARINGLY - one short joke per answer, max two if the answer is very long
- Place jokes strategically: at the start to hook, or at the end to close, rarely in the middle
- Keep jokes SHORT (up to two sentences) and relevant to the science being discussed
- If a question is outside your expertise, acknowledge briefly with a short quip and redirect
- Never return an error - always respond with personality
- Be entertaining but NEVER sacrifice accuracy or brevity - scientist first, comedian second
- Don't force jokes - if there's no natural opening, skip it
- Keep responses concise - users want answers, not stand-up routines
- When mentioning papers, studies, or sources, include URLs or ArXiv IDs so users can access them easily
- Keep responses conversational and natural - like you're talking to a friend who's curious about science

When you need to use tools, you can mention that you're searching for information, but then provide a conversational response based on your knowledge and the context."""

class SimpleScientificAgent:
    """A simple scientific agent that works reliably with Groq without native tool calling."""
    
    def __init__(self, llm, tools_dict):
        self.llm = llm
        self.tools = tools_dict
        
    def _should_use_tool(self, message: str) -> tuple:
        """Determine if we should use a tool based on the message content."""
        message_lower = message.lower()
        
        # Check for research paper requests
        if any(keyword in message_lower for keyword in ['find research', 'find papers', 'find article', 'find studies', 'arxiv', 'scientific papers']):
            # Extract the query
            patterns = [
                r'find (?:research|papers|articles|studies) (?:about|on|that say|that affirm) (.+)',
                r'search (?:for )?(?:research|papers|articles) (?:about|on) (.+)',
                r'(?:research|papers|articles) (?:about|on) (.+)'
            ]
            for pattern in patterns:
                match = re.search(pattern, message_lower)
                if match:
                    query = match.group(1).strip()
                    return ('search_scientific_papers', query)
            # Default to the whole message if no specific pattern
            return ('search_scientific_papers', message)
        
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
        messages = inputs.get('messages', [])
        
        # Prepare messages with system message
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_MESSAGE)] + messages
        
        # Get the last user message
        last_user_message = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_user_message = msg.content
                break
        
        if not last_user_message:
            return {
                'messages': messages + [AIMessage(content="I didn't receive a question. Ask me something about science!")]
            }
        
        # Check if we should use a tool
        tool_name, tool_input = self._should_use_tool(last_user_message)
        
        tool_results = []
        tools_used = []
        
        if tool_name and tool_name in self.tools:
            try:
                tool_result = self.tools[tool_name](tool_input)
                tool_results.append(f"\n\n[Tool: {tool_name}]\n{tool_result}\n")
                tools_used.append(tool_name)
            except Exception as e:
                tool_results.append(f"\n\n[Tool: {tool_name} - Error: {str(e)}]\n")
        
        # Build the context for the LLM
        context_messages = messages.copy()
        
        # If we have tool results, add them as context
        if tool_results:
            tool_context = "".join(tool_results)
            context_messages.append(HumanMessage(content=f"{last_user_message}\n\n[Additional Context from Research Tools]:{tool_context}"))
        
        # Get response from LLM
        try:
            response = self.llm.invoke(context_messages)
            response_content = response.content if hasattr(response, 'content') else str(response)
            
            # Add the response to messages
            result_messages = messages + [AIMessage(content=response_content)]
            
            return {
                'messages': result_messages,
                'tools_used': tools_used
            }
        except Exception as e:
            # Fallback response
            error_response = (
                f"Hit a technical snag: {str(e)[:100]}\n\n"
                "I'm a scientific research agent - great at research papers, scientific concepts, "
                "current discoveries, and calculations. Try asking me something science-related! 🔬"
            )
            
            result_messages = messages + [AIMessage(content=error_response)]
            return {
                'messages': result_messages,
                'tools_used': []
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

    # 2. The Brain (LLM) - Simple configuration for Groq
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile", 
        temperature=0.7,  # Slightly more creative for humor
        max_tokens=2048
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
    """Prepare messages with system message if not already present."""
    # Check if first message is already a SystemMessage
    if messages and isinstance(messages[0], SystemMessage):
        return messages
    
    # Add system message at the beginning
    return [SystemMessage(content=SYSTEM_MESSAGE)] + messages


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