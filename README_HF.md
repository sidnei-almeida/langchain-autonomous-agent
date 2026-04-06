<div align="center">

## Hugging Face Spaces — deployment

**Docker SDK · FastAPI · Uvicorn**

*Companion guide for [Heisenberg — Autonomous Scientific Agent](https://github.com/sidnei-almeida/langchain-autonomous-agent).*

[![Docker](https://img.shields.io/badge/SDK-Docker-2496ED.svg?logo=docker&logoColor=white)](https://huggingface.co/docs/hub/spaces-sdks-embed)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[**Main repository**](https://github.com/sidnei-almeida/langchain-autonomous-agent) · Maintainer: [**@sidnei-almeida**](https://github.com/sidnei-almeida)

</div>

---

This project runs on [Hugging Face Spaces](https://huggingface.co/spaces) using the **Docker** SDK. The container serves a FastAPI application via Uvicorn (see `app.py` and `Dockerfile`).

---

## Requirements

- A Hugging Face account with permission to create Spaces
- A Groq API key ([console.groq.com](https://console.groq.com))

---

## Creating the Space

1. Open [huggingface.co/new-space](https://huggingface.co/new-space).
2. Choose a name and select **Docker** as the SDK.
3. Create the Space from this repository (Git integration) or push the repository contents to the Space Git remote.

---

## Secrets

1. Open the Space **Settings** page.
2. Under **Secrets and variables**, add:

| Name | Value |
|------|--------|
| `GROQ_API_KEY` | Your Groq API key |

The application reads this variable at runtime; a local `.env` file is not required on the Space.

---

## Repository layout

The Space repository should include at minimum:

| File | Role |
|------|------|
| `Dockerfile` | Image build instructions |
| `requirements.txt` | Python dependencies |
| `app.py` | Uvicorn entry point |
| `api.py` | FastAPI application |
| `agent.py` | Agent and tools |
| `README.md` | Space card metadata (YAML front matter supported) |

---

## Local verification

Before pushing, validate locally:

```bash
export GROQ_API_KEY="your_key_here"
python app.py
```

Visit `http://localhost:7860/docs`.

---

## Build and runtime

- The Space builder runs `docker build` using the repository `Dockerfile`.
- First startup may take several minutes while dependencies install.
- LLM calls are subject to Groq rate limits and queueing; allow adequate request timeouts on clients.

---

## Troubleshooting

| Issue | Action |
|-------|--------|
| Build fails | Confirm `requirements.txt` pins are installable; check build logs |
| `GROQ_API_KEY` errors | Verify the secret name matches exactly and redeploy |
| Import errors | Ensure `api.py` and `agent.py` sit beside `app.py` in the Space root |
| Timeouts | Increase client timeout; Groq latency varies with load |

---

## Documentation

- [README.md](README.md) — project overview
- [API_DOCS.md](API_DOCS.md) — REST API reference
- [DOCKER.md](DOCKER.md) — container operations
