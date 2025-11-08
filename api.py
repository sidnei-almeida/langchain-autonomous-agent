"""
FastAPI REST API for Scientific Research Agent
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import os
from dotenv import load_dotenv
from agent import create_scientific_agent, prepare_messages
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Scientific Research Agent API",
    description="An autonomous AI agent specialized in scientific research with access to multiple tools",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agent instance
agent = None

def get_agent():
    """Get or create the agent instance."""
    global agent
    if agent is None:
        try:
            agent = create_scientific_agent()
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize agent: {str(e)}"
            )
    return agent

# Request/Response Models
class QueryRequest(BaseModel):
    question: str = Field(..., description="The scientific question to ask", min_length=1)
    include_history: bool = Field(default=False, description="Include conversation history in response")

class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")

class QueryResponse(BaseModel):
    answer: str = Field(..., description="The agent's answer")
    question: str = Field(..., description="The original question")
    tools_used: Optional[List[str]] = Field(default=None, description="List of tools used by the agent")
    processing_time: Optional[float] = Field(default=None, description="Processing time in seconds")

class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="List of chat messages")
    
class ChatResponse(BaseModel):
    message: ChatMessage = Field(..., description="The assistant's response message")
    tools_used: Optional[List[str]] = Field(default=None, description="List of tools used")
    processing_time: Optional[float] = Field(default=None, description="Processing time in seconds")

class HealthResponse(BaseModel):
    status: str = Field(..., description="API status")
    agent_initialized: bool = Field(..., description="Whether the agent is initialized")
    available_tools: List[str] = Field(..., description="List of available tools")

# Endpoints
@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Scientific Research Agent API",
        "version": "1.0.0",
        "description": "An autonomous AI agent specialized in scientific research",
        "endpoints": {
            "health": "/health",
            "query": "/api/query",
            "chat": "/api/chat",
            "tools": "/api/tools",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint."""
    global agent
    try:
        if agent is None:
            # Try to initialize
            agent = create_scientific_agent()
        
        available_tools = [
            "Web Search (DuckDuckGo)",
            "Wikipedia",
            "ArXiv",
            "Scientific Calculator"
        ]
        
        return HealthResponse(
            status="healthy",
            agent_initialized=agent is not None,
            available_tools=available_tools
        )
    except Exception as e:
        return HealthResponse(
            status=f"unhealthy: {str(e)}",
            agent_initialized=False,
            available_tools=[]
        )

@app.get("/api/tools", tags=["Agent"])
async def get_tools():
    """Get list of available tools."""
    return {
        "tools": [
            {
                "name": "Web Search",
                "provider": "DuckDuckGo",
                "description": "Searches for up-to-date information on the internet. Use for news, recent events, and general information."
            },
            {
                "name": "Wikipedia",
                "provider": "Wikipedia API",
                "description": "Searches for detailed and encyclopedic information. Ideal for concepts, biographies, historical events, and in-depth explanations."
            },
            {
                "name": "ArXiv",
                "provider": "ArXiv API",
                "description": "Searches and retrieves scientific articles. Use to find academic papers, recent research, and scientific literature."
            },
            {
                "name": "Scientific Calculator",
                "provider": "Custom",
                "description": "Performs complex mathematical calculations including trigonometric, logarithmic, and exponential functions."
            }
        ]
    }

@app.post("/api/query", response_model=QueryResponse, tags=["Agent"])
async def query_agent(request: QueryRequest):
    """
    Query the scientific agent with a single question.
    
    This endpoint processes a single question and returns the agent's answer.
    For multi-turn conversations, use the /api/chat endpoint instead.
    """
    import time
    start_time = time.time()
    
    try:
        agent = get_agent()
        
        # Prepare messages with system message
        agent_messages = prepare_messages([HumanMessage(content=request.question)])
        
        # Invoke the agent with LangChain message objects
        result = agent.invoke({"messages": agent_messages})
        messages = result.get("messages", [])
        
        # Extract the final answer
        final_answer = next(
            (msg.content for msg in reversed(messages) if isinstance(msg, AIMessage)),
            None,
        )
        
        if not final_answer:
            # Instead of raising an error, return a funny, friendly response
            return QueryResponse(
                answer=(
                    "Okay, so I'm scratching my head here because I couldn't quite nail down a complete "
                    "answer for you. But hey, that's science for you - sometimes things don't work out perfectly!\n\n"
                    "Here's the deal: I'm a scientific research agent, which means I'm your go-to guy for "
                    "science, research, math, and all that nerdy stuff we love. If you're asking me about "
                    "something that's way outside my wheelhouse, I might not be the best choice.\n\n"
                    "But here's what I CAN do - I've got these amazing research tools (ArXiv, Wikipedia, "
                    "web search, and a calculator that's probably doing calculus in its sleep). Ask me "
                    "something science-related and let's see what kind of knowledge we can dig up together! "
                    "Physics, chemistry, biology, math, computer science, recent discoveries - I'm all about it! 🔬"
                ),
                question=request.question,
                tools_used=None,
                processing_time=round(time.time() - start_time, 2)
            )
        
        # Extract tools used (if available in the result)
        tools_used = []
        if "messages" in result:
            # Check for tool messages
            for msg in messages:
                if hasattr(msg, 'name') and msg.name:
                    # This is a tool message
                    tools_used.append(msg.name)
                elif hasattr(msg, 'tool_calls') and msg.tool_calls:
                    # This is an AI message with tool calls
                    tools_used.extend([tc.get('name', 'unknown') for tc in msg.tool_calls])
        
        # Remove duplicates and clean up
        tools_used = list(set(tools_used)) if tools_used else None
        
        processing_time = time.time() - start_time
        
        return QueryResponse(
            answer=final_answer,
            question=request.question,
            tools_used=tools_used,
            processing_time=round(processing_time, 2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        # Log the error for debugging
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error processing query: {str(e)}")
        print(f"Traceback: {error_trace}")
        
        # Return a friendly, funny error response instead of crashing
        return QueryResponse(
            answer=(
                "Alright, so here's the thing - I hit a little snag trying to process that question. "
                "But don't worry, I'm still here and ready to help!\n\n"
                "I'm a scientific research agent, which means I'm really good at science stuff - "
                "physics, chemistry, math, research papers, all that good stuff. But if you're asking "
                "me about something way outside my lane, I might not be your guy.\n\n"
                "Try asking me something science-related and watch me work my magic with my research "
                "tools. I've got ArXiv, Wikipedia, web search, and a calculator that's probably smarter "
                "than most of us. Let's do this! 🔬✨"
            ),
            question=request.question,
            tools_used=None,
            processing_time=round(time.time() - start_time, 2)
        )

@app.post("/api/chat", response_model=ChatResponse, tags=["Agent"])
async def chat_with_agent(request: ChatRequest):
    """
    Chat with the agent using conversation history.
    
    This endpoint supports multi-turn conversations by maintaining context
    through the message history.
    """
    import time
    start_time = time.time()
    
    try:
        agent = get_agent()
        
        # Convert messages to LangChain message objects
        agent_messages = []
        for msg in request.messages:
            if msg.role == "user":
                agent_messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                agent_messages.append(AIMessage(content=msg.content))
            elif msg.role == "system":
                agent_messages.append(SystemMessage(content=msg.content))
        
        # Prepare messages with system message if not already present
        agent_messages = prepare_messages(agent_messages)
        
        # Invoke the agent with full conversation history
        result = agent.invoke({"messages": agent_messages})
        messages = result.get("messages", [])
        
        # Extract the final answer (last AI message)
        final_answer = next(
            (msg.content for msg in reversed(messages) if isinstance(msg, AIMessage)),
            None,
        )
        
        if not final_answer:
            # Instead of raising an error, return a funny, friendly response
            return ChatResponse(
                message=ChatMessage(
                    role="assistant",
                    content=(
                        "Alright, so here's what happened - I couldn't quite put together a complete response "
                        "for you. But don't worry, I'm not giving up on you!\n\n"
                        "I'm a scientific research agent, which basically means I'm really good at the science "
                        "stuff - physics, chemistry, biology, math, research papers, you name it. But if you're "
                        "asking me about something that's way outside my expertise (like pop culture or random "
                        "stuff), I might not be your best bet.\n\n"
                        "Here's what I CAN do though - I've got these awesome research tools at my disposal: "
                        "ArXiv for scientific papers, Wikipedia for deep dives, web search for current info, "
                        "and a calculator that's probably solving equations in its dreams.\n\n"
                        "Try asking me something science-related and let's see what kind of knowledge we can "
                        "uncover together! Physics, chemistry, biology, math, computer science, recent "
                        "discoveries - that's my jam! 🔬✨"
                    )
                ),
                tools_used=None,
                processing_time=round(time.time() - start_time, 2)
            )
        
        # Extract tools used
        tools_used = []
        if "messages" in result:
            for msg in messages:
                if hasattr(msg, 'name') and msg.name:
                    # This is a tool message
                    tools_used.append(msg.name)
                elif hasattr(msg, 'tool_calls') and msg.tool_calls:
                    # This is an AI message with tool calls
                    tools_used.extend([tc.get('name', 'unknown') for tc in msg.tool_calls])
        
        # Remove duplicates and clean up
        tools_used = list(set(tools_used)) if tools_used else None
        
        processing_time = time.time() - start_time
        
        return ChatResponse(
            message=ChatMessage(role="assistant", content=final_answer),
            tools_used=tools_used,
            processing_time=round(processing_time, 2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        # Log the error for debugging
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error processing chat: {str(e)}")
        print(f"Traceback: {error_trace}")
        
        # Return a friendly error response instead of crashing
        # Get the last user message for context
        last_user_message = next(
            (msg.content for msg in reversed(request.messages) if msg.role == "user"),
            "your question"
        )
        
        return ChatResponse(
            message=ChatMessage(
                role="assistant",
                content=(
                    "Okay, so I hit a little bump in the road trying to process that question. "
                    "But hey, that's life - sometimes things don't go exactly as planned!\n\n"
                    "Here's the deal: I'm a scientific research agent, which means I'm really good "
                    "at the science stuff - physics, chemistry, biology, math, research papers, all "
                    "that good stuff. But if you're asking me about something way outside my lane "
                    "(like pop culture, entertainment, or random life stuff), I might not be your guy.\n\n"
                    "But here's what I CAN do - I've got these amazing research tools: ArXiv for "
                    "scientific papers, Wikipedia for deep knowledge, web search for current info, "
                    "and a calculator that's probably doing math in its sleep.\n\n"
                    "Try asking me something science-related and let's see what kind of knowledge we "
                    "can dig up together! Physics, chemistry, biology, math, computer science, recent "
                    "discoveries - that's where I shine! 🔬✨"
                )
            ),
            tools_used=None,
            processing_time=round(time.time() - start_time, 2)
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

