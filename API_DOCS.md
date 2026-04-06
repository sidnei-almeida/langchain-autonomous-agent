<div align="center">

## REST API reference

**FastAPI · OpenAPI**

*Heisenberg — Autonomous Scientific Agent*

[![OpenAPI](https://img.shields.io/badge/OpenAPI-3.0-6BA539.svg?logo=openapiinitiative)](https://swagger.io/specification/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[**Repository**](https://github.com/sidnei-almeida/langchain-autonomous-agent) · Maintainer: [**@sidnei-almeida**](https://github.com/sidnei-almeida)

</div>

---

This document describes the HTTP API exposed by the FastAPI application (`api.py`). For interactive exploration, use the OpenAPI UI at `/docs` on a running instance.

---

## Base URLs

- **Local development:** `http://localhost:7860`
- **Hugging Face Spaces (example):** `https://huggingface.co/spaces/salmeida/langchain-agent` — append paths as needed; replace the namespace or Space name if yours differs.

---

## Endpoints

### GET `/`

Returns service metadata and a map of available routes.

**Response (example)**

```json
{
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
```

---

### GET `/health`

Reports process health and whether the agent singleton has been initialized.

**Response (example)**

```json
{
  "status": "healthy",
  "agent_initialized": true,
  "available_tools": [
    "Web Search (DuckDuckGo)",
    "Wikipedia",
    "ArXiv",
    "Scientific Calculator"
  ]
}
```

On initialization failure, `status` may describe the error and `available_tools` may be empty.

---

### GET `/api/tools`

Returns a static catalog of integrated tools with human-readable descriptions.

**Response (shape)**

```json
{
  "tools": [
    {
      "name": "Web Search",
      "provider": "DuckDuckGo",
      "description": "..."
    }
  ]
}
```

---

### POST `/api/query`

Single-turn query. The server builds a fresh message list (with system prompt), invokes the agent once, and returns the final assistant message plus metadata.

**Request body**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `question` | string | Yes | User question (minimum length 1) |
| `include_history` | boolean | No | Reserved; default `false` |

```json
{
  "question": "What are the latest advances in quantum computing?",
  "include_history": false
}
```

**Response body**

| Field | Type | Description |
|-------|------|-------------|
| `answer` | string | Final assistant text |
| `question` | string | Echo of the submitted question |
| `tools_used` | string[] \| null | Tool identifiers used during that turn, if any |
| `processing_time` | number \| null | Elapsed seconds (two decimal places typical) |
| `structured` | object \| null | Optional extraction of URLs, arXiv-style links, DOIs, and inferred authors |

**cURL**

```bash
curl -s -X POST "http://localhost:7860/api/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the latest advances in quantum computing?"}'
```

---

### POST `/api/chat`

Multi-turn conversation. Messages are converted to LangChain `HumanMessage` / `AIMessage` / `SystemMessage` objects before invocation.

**Request body**

```json
{
  "messages": [
    { "role": "user", "content": "What is quantum entanglement?" },
    { "role": "assistant", "content": "Quantum entanglement is ..." },
    { "role": "user", "content": "Can you give me an example?" }
  ]
}
```

Supported `role` values: `user`, `assistant`, `system`.

**Response body**

| Field | Type | Description |
|-------|------|-------------|
| `message` | object | `{ "role": "assistant", "content": "..." }` |
| `tools_used` | string[] \| null | Tools used for the last generation |
| `processing_time` | number \| null | Elapsed seconds |
| `structured` | object \| null | Same extraction semantics as `/api/query` |

**cURL**

```bash
curl -s -X POST "http://localhost:7860/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is quantum entanglement?"}]}'
```

---

## Interactive documentation

| UI | Path |
|----|------|
| Swagger UI | `/docs` |
| ReDoc | `/redoc` |

---

## Error handling

Validation failures typically yield **HTTP 422** with a FastAPI validation detail payload.

Application logic may return **HTTP 200** with a fallback `answer` string when the model produces no content or when an internal exception is caught (see implementation in `api.py`). For strict error semantics, adapt the handlers in your fork.

---

## Rate limiting and authentication

The reference deployment does not enforce rate limits or API-key authentication. For production, place the service behind a reverse proxy or API gateway with throttling and credential checks.

---

## Client examples

**Python**

```python
import requests

r = requests.post(
    "http://localhost:7860/api/query",
    json={"question": "What is machine learning?"},
    timeout=120,
)
print(r.json())

r2 = requests.post(
    "http://localhost:7860/api/chat",
    json={"messages": [{"role": "user", "content": "Explain quantum computing"}]},
    timeout=120,
)
print(r2.json())
```

**JavaScript (fetch)**

```javascript
const res = await fetch('http://localhost:7860/api/query', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ question: 'What is machine learning?' }),
});
console.log(await res.json());
```

---

## Versioning

The API version string returned by `GET /` is defined in `api.py` (`FastAPI(..., version="1.0.0")`). Increment it when you introduce breaking changes.
