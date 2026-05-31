"""
FastAPI REST API — Gray Matter Research Agent
"""
from __future__ import annotations

import logging
import os
import time
import traceback

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Any

from agent import create_scientific_agent, prepare_messages
from api_config import APIKeyMiddleware, RequestSizeLimitMiddleware, get_cors_origins
from api_helpers import build_enriched_response

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gray_matter.api")

app = FastAPI(
    title="Gray Matter Research Agent API",
    description=(
        "Agentic scientific research API: plans, searches, ranks evidence, "
        "and synthesizes grounded answers using Groq, arXiv, Wikipedia, and web search."
    ),
    version="3.0.0",
)

app.add_middleware(RequestSizeLimitMiddleware)
app.add_middleware(
    APIKeyMiddleware,
    api_key=os.getenv("GRAY_MATTER_API_KEY") or None,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = None


def get_agent():
    global agent
    if agent is None:
        try:
            agent = create_scientific_agent()
        except Exception as e:
            logger.error("Agent init failed: %s", e)
            raise HTTPException(status_code=500, detail="Agent initialization failed.")
    return agent


# --- Models ---

class LegacyStructured(BaseModel):
    sources: Optional[List[str]] = None
    authors: Optional[List[str]] = None


class SourceItem(BaseModel):
    title: str = ""
    url: str = ""
    source_type: str = ""
    snippet: str = ""
    relevance_score: float = 0.0
    used_in_answer: bool = False


class PaperItem(BaseModel):
    title: str = ""
    authors: Optional[List[str]] = None
    year: Optional[int] = None
    url: str = ""
    pdfUrl: Optional[str] = None
    summary: Optional[str] = None
    categories: Optional[List[str]] = None
    relevanceScore: Optional[int] = None
    whyItMatches: Optional[str] = None


class EnrichedResponse(BaseModel):
    answer: str
    tools_used: Optional[List[str]] = None
    processing_time: Optional[float] = None
    intent: Optional[str] = None
    research_plan: Optional[List[str]] = None
    sources: Optional[List[Any]] = None
    papers: Optional[List[Any]] = None
    confidence: Optional[float] = None
    limitations: Optional[List[str]] = None
    follow_up_questions: Optional[List[str]] = None
    structured: Optional[LegacyStructured] = None


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=8000)
    include_history: bool = False


class QueryResponse(EnrichedResponse):
    question: str


class ChatMessage(BaseModel):
    role: str
    content: str = Field(..., max_length=8000)


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., max_length=50)


class ChatResponse(BaseModel):
    message: ChatMessage
    tools_used: Optional[List[str]] = None
    processing_time: Optional[float] = None
    intent: Optional[str] = None
    research_plan: Optional[List[str]] = None
    sources: Optional[List[Any]] = None
    papers: Optional[List[Any]] = None
    confidence: Optional[float] = None
    limitations: Optional[List[str]] = None
    follow_up_questions: Optional[List[str]] = None
    structured: Optional[LegacyStructured] = None


class ResearchRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=8000)
    depth: Literal["quick", "standard", "deep"] = "standard"
    max_sources: int = Field(default=8, ge=1, le=20)


class ResearchResponse(EnrichedResponse):
    question: str
    depth: str


class HealthResponse(BaseModel):
    status: str
    agent_initialized: bool
    available_tools: List[str]


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled error: %s\n%s", exc, traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred. Please retry later."},
    )


SPACE_PUBLIC_URL = os.getenv(
    "SPACE_PUBLIC_URL",
    "https://salmeida-langchain-agent.hf.space",
)


@app.get("/", tags=["General"])
async def root(request: Request):
    """JSON for API clients; simple HTML landing for browser / HF Space App tab."""
    payload = {
        "name": "Gray Matter Research Agent API",
        "version": "3.0.0",
        "space_url": SPACE_PUBLIC_URL.rstrip("/"),
        "endpoints": {
            "health": "/health",
            "query": "/api/query",
            "chat": "/api/chat",
            "research": "/api/research",
            "tools": "/api/tools",
            "docs": "/docs",
        },
        "frontend_hint": {
            "base_url": SPACE_PUBLIC_URL.rstrip("/"),
            "chat": "POST /api/chat",
            "query": "POST /api/query",
            "deep_research": "POST /api/research",
        },
    }
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        base = SPACE_PUBLIC_URL.rstrip("/")
        html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>Gray Matter Research Agent</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 720px; margin: 2rem auto; padding: 0 1rem;
    background: #0d1117; color: #e6edf3; }}
  a {{ color: #58a6ff; }} h1 {{ font-weight: 600; }}
  code {{ background: #161b22; padding: 2px 6px; border-radius: 4px; }}
  .badge {{ background: #238636; color: #fff; padding: 2px 8px; border-radius: 12px; font-size: 0.85rem; }}
</style></head><body>
<h1>Gray Matter Research Agent <span class="badge">v3.0.0</span></h1>
<p>Agentic scientific research API — plan, search, rank evidence, synthesize.</p>
<p><strong>Base URL:</strong> <code>{base}</code></p>
<ul>
  <li><a href="/docs">Swagger UI</a> — interactive API</li>
  <li><code>POST /api/query</code> — single question</li>
  <li><code>POST /api/chat</code> — multi-turn chat</li>
  <li><code>POST /api/research</code> — deep research mode</li>
</ul>
<p>Same Hugging Face Space for backend + API. Point your Gray Matter LABS frontend to this base URL.</p>
</body></html>"""
        return HTMLResponse(html)
    return payload


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    global agent
    try:
        if agent is None:
            agent = create_scientific_agent()
        return HealthResponse(
            status="healthy",
            agent_initialized=True,
            available_tools=["Web Search", "Wikipedia", "ArXiv", "Calculator"],
        )
    except Exception as e:
        return HealthResponse(
            status=f"unhealthy: {str(e)[:120]}",
            agent_initialized=False,
            available_tools=[],
        )


@app.get("/api/tools", tags=["Agent"])
async def get_tools():
    return {
        "tools": [
            {"name": "Web Search", "provider": "DuckDuckGo",
             "description": "Current web context and news."},
            {"name": "Wikipedia", "provider": "Wikipedia API",
             "description": "Encyclopedic background."},
            {"name": "ArXiv", "provider": "ArXiv API",
             "description": "Ranked scientific papers with relevance scoring."},
            {"name": "Calculator", "provider": "Custom",
             "description": "Deterministic math evaluation."},
        ]
    }


def _run_agent(question: str, messages=None, depth="standard", max_sources=8):
    ag = get_agent()
    if messages is not None:
        prepared = prepare_messages(messages)
    else:
        prepared = prepare_messages([HumanMessage(content=question)])
    return ag.invoke({
        "messages": prepared,
        "depth": depth,
        "max_sources": max_sources,
    })


@app.post("/api/query", response_model=QueryResponse, tags=["Agent"])
async def query_agent(request: QueryRequest):
    start = time.time()
    try:
        result = _run_agent(request.question)
        payload = build_enriched_response(result, question=request.question)
        payload["processing_time"] = round(time.time() - start, 2)
        return QueryResponse(**payload)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Query error: %s", e)
        return QueryResponse(
            answer="Unable to process your question. Please try again with a clearer query.",
            question=request.question,
            processing_time=round(time.time() - start, 2),
            confidence=0.0,
            limitations=["Request processing failed."],
        )


@app.post("/api/chat", response_model=ChatResponse, tags=["Agent"])
async def chat_with_agent(request: ChatRequest):
    start = time.time()
    try:
        agent_messages = []
        for msg in request.messages:
            if msg.role == "user":
                agent_messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                agent_messages.append(AIMessage(content=msg.content))

        prepared = prepare_messages(agent_messages)
        last_user = next(
            (m.content for m in reversed(prepared) if isinstance(m, HumanMessage)),
            "",
        )
        result = _run_agent(last_user, messages=prepared)
        payload = build_enriched_response(result)
        payload["processing_time"] = round(time.time() - start, 2)

        return ChatResponse(
            message=ChatMessage(role="assistant", content=payload["answer"]),
            tools_used=payload.get("tools_used"),
            intent=payload.get("intent"),
            research_plan=payload.get("research_plan"),
            sources=payload.get("sources"),
            papers=payload.get("papers"),
            confidence=payload.get("confidence"),
            limitations=payload.get("limitations"),
            follow_up_questions=payload.get("follow_up_questions"),
            structured=LegacyStructured(**payload["structured"])
            if payload.get("structured") else None,
            processing_time=payload["processing_time"],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Chat error: %s", e)
        return ChatResponse(
            message=ChatMessage(
                role="assistant",
                content="Request failed. Please retry with a science or research question.",
            ),
            processing_time=round(time.time() - start, 2),
            confidence=0.0,
        )


@app.post("/api/research", response_model=ResearchResponse, tags=["Agent"])
async def deep_research(request: ResearchRequest):
    start = time.time()
    try:
        depth = request.depth
        max_sources = request.max_sources
        if depth == "deep":
            max_sources = max(max_sources, 12)

        result = _run_agent(
            request.question,
            depth=depth,
            max_sources=max_sources,
        )
        payload = build_enriched_response(result, question=request.question)
        payload["processing_time"] = round(time.time() - start, 2)
        payload["depth"] = depth
        return ResearchResponse(**payload)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Research error: %s", e)
        return ResearchResponse(
            answer="Deep research request failed. Try a more specific question.",
            question=request.question,
            depth=request.depth,
            processing_time=round(time.time() - start, 2),
            confidence=0.0,
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
