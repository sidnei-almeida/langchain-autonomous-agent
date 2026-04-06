"""
FastAPI REST API for Scientific Research Agent
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import os
import json
import re
import shutil
from pathlib import Path
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

class StructuredResponse(BaseModel):
    sources: Optional[List[str]] = Field(default=None, description="URLs and references mentioned in the response")
    authors: Optional[List[str]] = Field(default=None, description="Authors mentioned in the response")

class QueryResponse(BaseModel):
    answer: str = Field(..., description="The agent's answer")
    question: str = Field(..., description="The original question")
    tools_used: Optional[List[str]] = Field(default=None, description="List of tools used by the agent")
    processing_time: Optional[float] = Field(default=None, description="Processing time in seconds")
    structured: Optional[StructuredResponse] = Field(default=None, description="Structured response data for frontend organization")

class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="List of chat messages")
    
class ChatResponse(BaseModel):
    message: ChatMessage = Field(..., description="The assistant's response message")
    tools_used: Optional[List[str]] = Field(default=None, description="List of tools used")
    processing_time: Optional[float] = Field(default=None, description="Processing time in seconds")
    structured: Optional[StructuredResponse] = Field(default=None, description="Structured response data for frontend organization")

class HealthResponse(BaseModel):
    status: str = Field(..., description="API status")
    agent_initialized: bool = Field(..., description="Whether the agent is initialized")
    available_tools: List[str] = Field(..., description="List of available tools")

# Helper function to extract sources/URLs and authors from response
def extract_structured_data(text: str) -> Optional[StructuredResponse]:
    """
    Extract URLs, references, and authors from the LLM response.
    Keep it simple - just sources and authors, nothing fancy.
    """
    try:
        sources = []
        authors = []
        
        # Extract URLs
        urls = re.findall(r'https?://[^\s\)]+', text)
        sources.extend(urls)
        
        # Extract ArXiv references (format: arXiv:1234.5678 or arXiv 1234.5678)
        arxiv_refs = re.findall(r'arXiv[:\s]+([0-9]+\.[0-9]+(?:v[0-9]+)?)', text, re.IGNORECASE)
        # Format ArXiv references as URLs
        for arxiv_id in arxiv_refs:
            sources.append(f"https://arxiv.org/abs/{arxiv_id}")
        
        # Extract DOI references
        dois = re.findall(r'doi[:\s/]+([0-9]+\.[0-9]+/[^\s\)]+)', text, re.IGNORECASE)
        for doi in dois:
            sources.append(f"https://doi.org/{doi}")
        
        # Extract authors (common patterns: "by Author Name", "Author et al.", "Authors: Name1, Name2")
        # Pattern 1: "by [Name]"
        by_authors = re.findall(r'\bby\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', text, re.IGNORECASE)
        authors.extend(by_authors)
        
        # Pattern 2: "Authors: Name1, Name2" or "Author: Name"
        author_pattern = re.findall(r'(?:Authors?|Written by|Published by)[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+et\s+al\.)?(?:\s*,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)*)', text, re.IGNORECASE)
        for match in author_pattern:
            # Split by comma and clean up
            author_list = [a.strip() for a in match.split(',')]
            authors.extend(author_list)
        
        # Pattern 3: "[Name] et al." format
        et_al_pattern = re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+et\s+al\.', text)
        authors.extend(et_al_pattern)
        
        # Remove duplicates and clean up
        sources = list(set(sources))
        authors = list(set([a.strip() for a in authors if len(a.strip()) > 2]))  # Filter out very short matches
        
        # Only return if we found something
        if sources or authors:
            return StructuredResponse(
                sources=sources[:20] if sources else None,  # Limit to 20 sources
                authors=authors[:10] if authors else None  # Limit to 10 authors
            )
        
    except Exception as e:
        print(f"Error extracting structured data: {str(e)}")
    
    return None

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
            return QueryResponse(
                answer=(
                    "No answer from the model that run. Ask something concrete — "
                    "science, math, a paper search, or a calculation. "
                    "Keep it short; I'll keep it shorter."
                ),
                question=request.question,
                tools_used=None,
                processing_time=round(time.time() - start_time, 2)
            )
        
        # Extract tools used - now from the custom agent result
        tools_used = result.get('tools_used', [])
        
        # Remove duplicates and clean up
        tools_used = list(set(tools_used)) if tools_used else None
        
        # Extract structured data from the response
        structured_data = extract_structured_data(final_answer)
        
        processing_time = time.time() - start_time
        
        return QueryResponse(
            answer=final_answer,
            question=request.question,
            tools_used=tools_used,
            processing_time=round(processing_time, 2),
            structured=structured_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        # Log the error for debugging
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error processing query: {str(e)}")
        print(f"Traceback: {error_trace}")
        
        return QueryResponse(
            answer=(
                "Something broke on the backend. Try again with a clear science or math question. "
                "If it keeps failing, check logs — not my problem to guess."
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
            return ChatResponse(
                message=ChatMessage(
                    role="assistant",
                    content=(
                        "Empty response. Ask a science or math question — one thing at a time."
                    )
                ),
                tools_used=None,
                processing_time=round(time.time() - start_time, 2)
            )
        
        # Extract tools used - now from the custom agent result
        tools_used = result.get('tools_used', [])
        
        # Remove duplicates and clean up
        tools_used = list(set(tools_used)) if tools_used else None
        
        # Extract structured data from the response
        structured_data = extract_structured_data(final_answer)
        
        processing_time = time.time() - start_time
        
        return ChatResponse(
            message=ChatMessage(role="assistant", content=final_answer),
            tools_used=tools_used,
            processing_time=round(processing_time, 2),
            structured=structured_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        # Log the error for debugging
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error processing chat: {str(e)}")
        print(f"Traceback: {error_trace}")
        
        return ChatResponse(
            message=ChatMessage(
                role="assistant",
                content=(
                    "Request failed. Retry with a science or math question — or check server logs."
                )
            ),
            tools_used=None,
            processing_time=round(time.time() - start_time, 2)
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

