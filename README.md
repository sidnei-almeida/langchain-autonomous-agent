---
title: Heisenberg — Autonomous Scientific Agent
emoji: ⚗️
colorFrom: gray
colorTo: indigo
sdk: docker
app_file: app.py
pinned: false
license: mit
---

# Heisenberg — Autonomous Scientific Agent

An autonomous AI agent that answers scientific questions with the precision and authority of Walter White — built on LangChain, Groq (Llama 3.3 70B), and a curated set of research tools.

The agent does not approximate. It calculates.

---

## Architecture

The agent (`SimpleScientificAgent`) intercepts each query, determines the optimal tool via keyword heuristics, executes it, injects the result as context, and passes the enriched message to the LLM for synthesis. The LLM never calls tools directly — the agent orchestrates everything.

```
User Query
    │
    ▼
Tool Selection (heuristic)
    │
    ├── Web Search (DuckDuckGo)
    ├── Wikipedia
    ├── ArXiv
    └── Calculator
    │
    ▼
Context Injection → LLM (Groq / Llama 3.3 70B)
    │
    ▼
Response
```

---

## Tools

| Tool | Purpose |
|---|---|
| DuckDuckGo Search | Real-time web information, news, recent events |
| Wikipedia | Encyclopedic knowledge, definitions, background |
| ArXiv | Academic papers — titles, abstracts, authors, PDF links |
| Scientific Calculator | Mathematical expressions, trigonometric, logarithmic, exponential functions |

---

## Stack

- **LangChain** + **LangChain-Groq** — agent framework and LLM integration
- **Groq API** — inference backend (Llama 3.3 70B Versatile)
- **FastAPI** + **Uvicorn** — REST API layer
- **DuckDuckGo Search**, **Wikipedia**, **ArXiv** — data sources
- **Docker** — containerized deployment

---

## Local Setup

**1. Clone the repository**
```bash
git clone https://github.com/sidnei-almeida/langchain-autonomous-agent.git
cd langchain-autonomous-agent
```

**2. Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Configure environment**
```bash
# Create a .env file
echo "GROQ_API_KEY=your_key_here" > .env
```

Get your Groq API key at [console.groq.com](https://console.groq.com).

---

## Usage

### REST API (default)

```bash
python app.py
```

The API will be available at `http://localhost:7860`. Interactive docs at `http://localhost:7860/docs`.

### CLI

```bash
# Interactive mode
python agent.py

# Single question
python agent.py "What are the latest advances in quantum computing?"
```

### Docker

```bash
# Using Docker Compose
docker-compose up --build

# Using Docker directly
docker build -t heisenberg-agent .
docker run -p 7860:7860 -e GROQ_API_KEY=your_key_here heisenberg-agent
```

---

## API Reference

### `POST /api/query`

Single-turn question with tool execution and structured response.

```json
{
  "question": "Find recent papers on neural scaling laws"
}
```

Response includes `answer`, `tools_used`, `processing_time`, and extracted `structured` data (URLs, ArXiv IDs, authors).

### `POST /api/chat`

Multi-turn conversation with message history.

```json
{
  "messages": [
    { "role": "user", "content": "What is entropy?" }
  ]
}
```

### `GET /health`

Returns service status and agent initialization state.

Full API documentation: [`API_DOCS.md`](API_DOCS.md)

---

## Deployment

### Hugging Face Spaces (Docker SDK)

This Space runs via Docker. To deploy your own instance:

1. Fork or clone this repository
2. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces) with **Docker** as the SDK
3. Push the repository to the Space's git remote
4. Add `GROQ_API_KEY` as a Secret in the Space settings (Settings → Secrets)

The Space will build automatically and serve the FastAPI application.

### Cloud Deployment (AWS / GCP / Azure)

```bash
# Build and push image
docker build -t heisenberg-agent .
docker tag heisenberg-agent your-registry/heisenberg-agent:latest
docker push your-registry/heisenberg-agent:latest
```

Set the `GROQ_API_KEY` environment variable in your cloud provider's secrets manager or container configuration.

---

## Project Structure

```
langchain-autonomous-agent/
├── agent.py            # Agent logic, tools, and persona
├── api.py              # FastAPI application and endpoints
├── app.py              # Entry point (Uvicorn server)
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container image definition
├── docker-compose.yml  # Local orchestration
├── API_DOCS.md         # Full API documentation
└── README.md           # This file
```

---

## License

MIT — open for educational and research use.
