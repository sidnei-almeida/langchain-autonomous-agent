# 🔬 Scientific Research Agent - API Documentation

## Base URL

```
http://localhost:7860  # Local development
https://huggingface.co/spaces/salmeida/my-scientific-agent  # Production
```

## Endpoints

### 1. Root - API Information

**GET** `/`

Returns basic API information and available endpoints.

**Response:**
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

### 2. Health Check

**GET** `/health`

Check API health status and agent initialization.

**Response:**
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

---

### 3. Get Available Tools

**GET** `/api/tools`

Get list of all available tools with descriptions.

**Response:**
```json
{
  "tools": [
    {
      "name": "Web Search",
      "provider": "DuckDuckGo",
      "description": "Searches for up-to-date information on the internet..."
    },
    {
      "name": "Wikipedia",
      "provider": "Wikipedia API",
      "description": "Searches for detailed and encyclopedic information..."
    },
    {
      "name": "ArXiv",
      "provider": "ArXiv API",
      "description": "Searches and retrieves scientific articles..."
    },
    {
      "name": "Scientific Calculator",
      "provider": "Custom",
      "description": "Performs complex mathematical calculations..."
    }
  ]
}
```

---

### 4. Query Agent (Single Question)

**POST** `/api/query`

Query the agent with a single question. Best for one-off queries.

**Request Body:**
```json
{
  "question": "What are the latest advances in quantum computing?",
  "include_history": false
}
```

**Response:**
```json
{
  "answer": "According to recent research on ArXiv...",
  "question": "What are the latest advances in quantum computing?",
  "tools_used": ["arxiv", "wikipedia"],
  "processing_time": 12.45
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:7860/api/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the latest advances in quantum computing?"
  }'
```

---

### 5. Chat with Agent (Multi-turn Conversation)

**POST** `/api/chat`

Chat with the agent using conversation history. Best for multi-turn conversations.

**Request Body:**
```json
{
  "messages": [
    {
      "role": "user",
      "content": "What is quantum entanglement?"
    },
    {
      "role": "assistant",
      "content": "Quantum entanglement is a phenomenon..."
    },
    {
      "role": "user",
      "content": "Can you give me an example?"
    }
  ]
}
```

**Response:**
```json
{
  "message": {
    "role": "assistant",
    "content": "Sure! Here's an example of quantum entanglement..."
  },
  "tools_used": ["wikipedia", "calculator"]
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:7860/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is quantum entanglement?"}
    ]
  }'
```

---

## Interactive API Documentation

FastAPI provides automatic interactive documentation:

- **Swagger UI**: `http://localhost:7860/docs`
- **ReDoc**: `http://localhost:7860/redoc`

## Error Responses

All endpoints may return the following error responses:

**400 Bad Request:**
```json
{
  "detail": "Validation error message"
}
```

**500 Internal Server Error:**
```json
{
  "detail": "Error processing query: error message"
}
```

## Rate Limiting

Currently, there are no rate limits implemented. Consider implementing rate limiting for production use.

## Authentication

Currently, no authentication is required. For production, consider adding API key authentication.

## Example Usage (Python)

```python
import requests

# Single query
response = requests.post(
    "http://localhost:7860/api/query",
    json={"question": "What is machine learning?"}
)
print(response.json())

# Chat conversation
response = requests.post(
    "http://localhost:7860/api/chat",
    json={
        "messages": [
            {"role": "user", "content": "Explain quantum computing"}
        ]
    }
)
print(response.json())
```

## Example Usage (JavaScript)

```javascript
// Single query
const response = await fetch('http://localhost:7860/api/query', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    question: 'What is machine learning?'
  })
});

const data = await response.json();
console.log(data);

// Chat conversation
const chatResponse = await fetch('http://localhost:7860/api/chat', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    messages: [
      { role: 'user', content: 'Explain quantum computing' }
    ]
  })
});

const chatData = await chatResponse.json();
console.log(chatData);
```

