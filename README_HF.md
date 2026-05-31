# Hugging Face Spaces — Gray Matter LABS

**Canonical Space (backend + API):**  
**https://huggingface.co/spaces/salmeida/langchain-agent**  
**Public URL:** **https://salmeida-langchain-agent.hf.space**

Use this **same Space** for everything — no second Space needed. Point the Gray Matter LABS frontend at:

```text
https://salmeida-langchain-agent.hf.space
```

Deploy this repository as a **Docker Space** serving the FastAPI research agent on port **7860**.

---

## 1. Create the Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2. **Space name** — e.g. `gray-matter-labs` or your org slug.
3. **SDK** — select **Docker** (not Gradio/Streamlit).
4. **Repository** — link this GitHub repo, or push files to the Space Git remote.

The Space reads configuration from **`README.md` front matter** at the repo root:

```yaml
sdk: docker
app_port: 7860
```

If the card does not update after push, open **Settings** and confirm **SDK = Docker** and **App port = 7860**.

---

## 2. Secret (required)

| Name | Value |
|------|--------|
| `GROQ_API_KEY` | Your key from [console.groq.com](https://console.groq.com) |

**Settings → Secrets and variables → New secret**

The app calls `load_dotenv()` but Spaces inject secrets as environment variables — no `.env` file is needed on the hub.

---

## 3. Files the builder must include

| File | Purpose |
|------|---------|
| `agent/` | Modular agent pipeline (graph, router, tools, …) |
| `arxiv_search.py` | Ranked arXiv search |
| `api.py`, `api_config.py`, `api_helpers.py` | FastAPI + middleware |
| `app.py` | Uvicorn entry |
| `Dockerfile` | HF Spaces image |

Do **not** commit `.env` or API keys.

---

## 4. Build & runtime

- HF runs `docker build` from `Dockerfile`.
- First build may take several minutes (`gcc` for some wheels).
- Default route: Space URL → API root; use **`/docs`** for Swagger UI.
- Health: `GET /health`

**Example request** (from browser or `curl`):

```bash
curl -X POST "https://YOUR_USER-YOUR_SPACE.hf.space/api/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "find papers about human-computer interaction"}'
```

---

## 5. Connect frontend (Gray Matter UI)

Point the frontend API base URL to the Space URL (no trailing slash):

```text
https://YOUR_USER-YOUR_SPACE.hf.space
```

Endpoints:

- `POST /api/query` — single turn
- `POST /api/chat` — history (`messages[]` with `user` / `assistant` only)

The backend filters UI mock messages and ranks arXiv results server-side.

---

## 6. Local check before push

```bash
export GROQ_API_KEY="your_key"
pip install -r requirements.txt
python app.py
# http://localhost:7860/docs
```

Or Docker:

```bash
docker build -t gray-matter-labs .
docker run -p 7860:7860 -e GROQ_API_KEY=your_key gray-matter-labs
```

---

## 7. Troubleshooting

| Issue | Fix |
|-------|-----|
| Build fails | Check build logs; verify `requirements.txt` installs on Python 3.11 |
| `GROQ_API_KEY not found` | Secret name must be exact; redeploy after adding |
| 502 / starting | Wait for cold start; check `/health` |
| Import error `arxiv_search` | Ensure `arxiv_search.py` is in repo root (included in `Dockerfile` COPY) |
| Wrong SDK | Space Settings → SDK **Docker**, not Gradio |
| Timeouts | arXiv fetch can take ~6–10s; increase client timeout |
| Empty paper list | Query may be ambiguous — API returns clarification instead of weak papers |

---

## 8. Updating an existing Space

1. Push to the connected branch (usually `main`).
2. HF rebuilds automatically.
3. Confirm **Secrets** still present after fork/duplicate.

---

## Related docs

- [README.md](./README.md) — project overview + HF front matter
- [API_DOCS.md](./API_DOCS.md) — REST schemas
- [DOCKER.md](./DOCKER.md) — generic Docker notes

**Maintainer:** [@sidnei-almeida](https://github.com/sidnei-almeida)
