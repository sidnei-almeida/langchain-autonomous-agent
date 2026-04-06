# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) where applicable.

---

## [Unreleased]

### Changed

- Documentation overhaul: professional README, API reference, Docker guide, Hugging Face notes, security advisory, and this changelog.
- Added full MIT `LICENSE` text.

---

## [1.0.0] — 2026-04-06

### Summary

Stable baseline for the Heisenberg scientific agent: FastAPI surface, Groq Llama 3.3 70B, heuristic tool routing, and Docker deployment.

### Architecture

- **Agent:** `SimpleScientificAgent` in `agent.py` selects tools by keyword heuristics, executes DuckDuckGo / Wikipedia / arXiv / calculator, injects results as context, then calls the chat model.
- **API:** `api.py` exposes `/`, `/health`, `/api/tools`, `/api/query`, `/api/chat` with optional structured extraction of URLs and authors from responses.
- **Entry:** `app.py` runs Uvicorn on `0.0.0.0` and port from `PORT` or `7860`.

### Persona

- System prompt models a Walter White / Heisenberg voice for scientific Q&A (fictional character; see disclaimer in README).

### Dependencies

- LangChain ecosystem, Groq, FastAPI, Uvicorn, community tools (DuckDuckGo, Wikipedia), arXiv client.

---

## Historical note

Earlier iterations experimented with LangGraph `create_react_agent` and different LangChain agent APIs. The current release uses the custom `SimpleScientificAgent` pattern for predictable behavior with the Groq stack and simplified tool orchestration.

---

Document history: prior LangGraph-oriented fix notes were consolidated into this changelog under **Historical note**.
