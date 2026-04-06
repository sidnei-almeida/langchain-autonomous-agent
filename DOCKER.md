<div align="center">

## Container deployment

**Docker · Docker Compose**

*Heisenberg — Autonomous Scientific Agent*

[![Docker](https://img.shields.io/badge/Docker-ready-2496ED.svg?logo=docker&logoColor=white)](https://docs.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[**Repository**](https://github.com/sidnei-almeida/langchain-autonomous-agent) · Maintainer: [**@sidnei-almeida**](https://github.com/sidnei-almeida)

</div>

---

This guide covers building and running the Heisenberg scientific agent with Docker and Docker Compose.

---

## Prerequisites

- Docker Engine ([installation](https://docs.docker.com/get-docker/))
- Docker Compose v2 (bundled with Docker Desktop on many systems)
- A valid `GROQ_API_KEY` from [console.groq.com](https://console.groq.com)

---

## Quick start — Docker Compose

**1. Provide the API key**

```bash
echo "GROQ_API_KEY=your_groq_api_key_here" > .env
```

**2. Build and start**

```bash
docker compose up --build
```

**3. Verify**

Open `http://localhost:7860/docs` or call `GET /health`.

**4. Stop**

```bash
docker compose down
```

---

## Quick start — Docker CLI

**Build**

```bash
docker build -t heisenberg-agent .
```

**Run**

```bash
docker run -d \
  --name heisenberg-agent \
  -p 7860:7860 \
  -e GROQ_API_KEY=your_groq_api_key_here \
  heisenberg-agent
```

**Logs**

```bash
docker logs -f heisenberg-agent
```

**Stop and remove**

```bash
docker stop heisenberg-agent
docker rm heisenberg-agent
```

---

## Production considerations

### Environment variables

Inject secrets via your orchestrator or host environment; avoid baking keys into images.

```bash
export GROQ_API_KEY=your_key_here
docker compose up -d
```

### Health checks

The FastAPI application exposes `GET /health`. Point container or load-balancer health checks at:

```
http://127.0.0.1:7860/health
```

Adjust the Dockerfile or Compose `healthcheck` if your runtime uses a different internal hostname.

### Example Compose overrides

You may add `restart: unless-stopped`, CPU and memory limits, and read-only root filesystems according to your security baseline.

---

## Cloud platforms

### AWS (ECS / Fargate)

1. Build and push to Amazon ECR.
2. Reference the image in a task definition.
3. Store `GROQ_API_KEY` in AWS Secrets Manager or SSM Parameter Store and inject as an environment variable.

### Google Cloud Run

Build with Cloud Build or Artifact Registry, deploy with `--set-secrets` or `--set-env-vars` for `GROQ_API_KEY`, and configure concurrency and timeouts for LLM latency.

### Azure Container Instances / Container Apps

Push to Azure Container Registry, deploy the image, and set environment variables from Azure Key Vault references.

---

## Troubleshooting

| Symptom | Check |
|---------|--------|
| Container exits immediately | `docker logs <container>`; verify `GROQ_API_KEY` is set |
| Port conflict | Map a different host port, e.g. `-p 8080:7860` |
| Slow first request | Agent initialization on first use; allow warm-up |
| Build failures | Network access for `pip install`; sufficient disk and memory |

---

## Image optimization

The default image uses `python:3.11-slim`. For smaller artifacts, consider multi-stage builds: install dependencies in a builder stage and copy only site-packages and application code into the final layer.

---

## Security practices

- Never commit `.env` files; use `.gitignore` and platform secrets.
- Run the process as a non-root user in production images when feasible.
- Scan images with your organization’s vulnerability workflow (`docker scout`, Trivy, etc.).
- Place the service behind TLS termination (reverse proxy or managed ingress).

---

## Monitoring

- `docker stats` for CPU and memory
- Application logs via `docker logs` or centralized logging driver
- External synthetic checks against `/health`

---

## Next steps

- Terminate TLS at nginx, Traefik, or a cloud load balancer
- Add authentication and rate limiting at the edge
- Define SLOs for latency and error rate on `/api/query` and `/api/chat`
