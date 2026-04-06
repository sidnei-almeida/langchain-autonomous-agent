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

## Overview

This repository provides a production-oriented REST API and CLI for scientific Q&A. The system combines heuristic tool routing (web search, Wikipedia, arXiv, calculator) with a Groq-hosted large language model. Responses are synthesized from retrieved context and structured metadata when applicable.

| Document | Description |
|----------|-------------|
| [API_DOCS.md](API_DOCS.md) | HTTP API specification and examples |
| [DOCKER.md](DOCKER.md) | Container build, deployment, and operations |
| [README_HF.md](README_HF.md) | Hugging Face Spaces deployment |
| [SECURITY_FIX.md](SECURITY_FIX.md) | Credential hygiene and incident response |
| [CHANGELOG.md](CHANGELOG.md) | Version history and implementation notes |
| [LICENSE](LICENSE) | MIT License (full text) |

---

## Architecture

The agent (`SimpleScientificAgent`) intercepts each query, determines the optimal tool via keyword heuristics, executes it, injects the result as context, and passes the enriched message to the LLM for synthesis. The LLM does not invoke tools directly; the application layer orchestrates tool use.

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
|------|---------|
| DuckDuckGo Search | Real-time web information, news, recent events |
| Wikipedia | Encyclopedic knowledge, definitions, background |
| ArXiv | Academic papers — titles, abstracts, authors, PDF links |
| Scientific Calculator | Mathematical expressions, trigonometric, logarithmic, exponential functions |

---

## Stack

- **LangChain** and **langchain-groq** — agent integration and Groq chat model
- **Groq API** — inference (Llama 3.3 70B Versatile)
- **FastAPI** and **Uvicorn** — REST API
- **DuckDuckGo Search**, **Wikipedia**, **ArXiv** — external data sources
- **Docker** — containerized deployment

---

## Prerequisites

- Python 3.11 or compatible
- A valid [Groq API key](https://console.groq.com)

---

## Local setup

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
echo "GROQ_API_KEY=your_key_here" > .env
```

---

## Usage

### REST API

```bash
python app.py
```

- Base URL: `http://localhost:7860`
- OpenAPI UI: `http://localhost:7860/docs`
- ReDoc: `http://localhost:7860/redoc`

### CLI

```bash
python agent.py
python agent.py "What are the latest advances in quantum computing?"
```

### Docker

```bash
docker-compose up --build
```

```bash
docker build -t heisenberg-agent .
docker run -p 7860:7860 -e GROQ_API_KEY=your_key_here heisenberg-agent
```

---

## API summary

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service metadata and endpoint index |
| GET | `/health` | Liveness and agent initialization |
| GET | `/api/tools` | Tool catalog |
| POST | `/api/query` | Single-turn question |
| POST | `/api/chat` | Multi-turn conversation |

See [API_DOCS.md](API_DOCS.md) for request and response schemas, curl examples, and client snippets.

---

## Deployment

### Hugging Face Spaces

Use the **Docker** SDK. Push this repository to your Space, set `GROQ_API_KEY` under **Settings → Secrets**, and allow the build to complete. See [README_HF.md](README_HF.md).

### Cloud (AWS, GCP, Azure)

Build the image, push to your registry, inject `GROQ_API_KEY` via the provider’s secret manager, and run the container behind your load balancer or ingress. See [DOCKER.md](DOCKER.md).

---

## Project structure

```
langchain-autonomous-agent/
├── agent.py            # Agent logic, tools, system persona
├── api.py              # FastAPI application
├── app.py              # Uvicorn entry point
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── LICENSE
├── API_DOCS.md
├── DOCKER.md
├── README_HF.md
├── SECURITY_FIX.md
├── CHANGELOG.md
└── README.md
```

---

## Security and operations

- Do not commit API keys. Use `.env` locally (ignored by Git) and platform secrets in production.
- The API ships with permissive CORS for development. Restrict `allow_origins` before public deployment.
- No built-in authentication or rate limiting; add reverse proxies, API keys, or OAuth as required.

See [SECURITY_FIX.md](SECURITY_FIX.md).

---

## Disclaimer

The conversational persona is inspired by a fictional character from *Breaking Bad*. This software is an independent educational and research project. It is not affiliated with AMC Networks, Sony Pictures Television, or the creators of *Breaking Bad*. Users remain responsible for compliance with applicable law and third-party terms of service (Groq, Wikipedia, arXiv, search providers).

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE).

---

## Maintainer

Sidnei Alves de Almeida
