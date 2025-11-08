# 🐳 Docker Deployment Guide

This guide explains how to deploy the Scientific Research Agent using Docker.

## Quick Start

### Prerequisites
- Docker installed ([Get Docker](https://docs.docker.com/get-docker/))
- Docker Compose installed (usually comes with Docker Desktop)
- Groq API key ([Get it here](https://console.groq.com))

### Option 1: Docker Compose (Recommended)

1. **Set up environment variables:**
   ```bash
   echo "GROQ_API_KEY=your_groq_api_key_here" > .env
   ```

2. **Build and run:**
   ```bash
   docker-compose up --build
   ```

3. **Access the app:**
   - Open your browser to `http://localhost:7860`

4. **Stop the container:**
   ```bash
   docker-compose down
   ```

### Option 2: Docker CLI

1. **Build the image:**
   ```bash
   docker build -t scientific-agent .
   ```

2. **Run the container:**
   ```bash
   docker run -d \
     --name scientific-agent \
     -p 7860:7860 \
     -e GROQ_API_KEY=your_groq_api_key_here \
     scientific-agent
   ```

3. **View logs:**
   ```bash
   docker logs -f scientific-agent
   ```

4. **Stop and remove:**
   ```bash
   docker stop scientific-agent
   docker rm scientific-agent
   ```

## Production Deployment

### Environment Variables

For production, use environment variables or secrets management:

```bash
# Using environment variable
export GROQ_API_KEY=your_key_here
docker-compose up -d

# Or pass directly
docker run -p 7860:7860 -e GROQ_API_KEY=your_key_here scientific-agent
```

### Docker Compose for Production

Edit `docker-compose.yml` for production settings:

```yaml
version: '3.8'

services:
  scientific-agent:
    build: .
    ports:
      - "7860:7860"
    environment:
      - GROQ_API_KEY=${GROQ_API_KEY}
    restart: always
    healthcheck:
      test: ["CMD", "python", "-c", "import requests; requests.get('http://localhost:7860')"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## Cloud Deployment

### AWS ECS / Fargate

1. Build and push to ECR:
   ```bash
   docker build -t scientific-agent .
   docker tag scientific-agent:latest your-account.dkr.ecr.region.amazonaws.com/scientific-agent:latest
   docker push your-account.dkr.ecr.region.amazonaws.com/scientific-agent:latest
   ```

2. Create ECS task definition with environment variable `GROQ_API_KEY`

### Google Cloud Run

```bash
# Build and push
gcloud builds submit --tag gcr.io/your-project/scientific-agent

# Deploy
gcloud run deploy scientific-agent \
  --image gcr.io/your-project/scientific-agent \
  --platform managed \
  --region us-central1 \
  --set-env-vars GROQ_API_KEY=your_key_here \
  --allow-unauthenticated
```

### Azure Container Instances

```bash
# Build and push to ACR
az acr build --registry your-registry --image scientific-agent:latest .

# Deploy
az container create \
  --resource-group your-resource-group \
  --name scientific-agent \
  --image your-registry.azurecr.io/scientific-agent:latest \
  --environment-variables GROQ_API_KEY=your_key_here \
  --ports 7860
```

## Troubleshooting

### Container won't start
- Check logs: `docker logs scientific-agent`
- Verify API key is set: `docker exec scientific-agent env | grep GROQ_API_KEY`

### Port already in use
- Change port mapping: `-p 8080:7860` (use port 8080 instead)
- Or stop the service using port 7860

### Build fails
- Check internet connection (needs to download dependencies)
- Verify Docker has enough resources (memory/disk)
- Try: `docker system prune` to free space

### Health check fails
- Wait longer for startup (agent initialization takes time)
- Check if Gradio is running: `docker exec scientific-agent ps aux | grep python`

## Image Size Optimization

The current Dockerfile uses Python 3.11 slim. For even smaller images:

```dockerfile
# Use multi-stage build
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY agent.py app.py .
ENV PATH=/root/.local/bin:$PATH
CMD ["python", "app.py"]
```

## Security Best Practices

1. **Never commit `.env` files** - Use secrets management
2. **Use non-root user** in production:
   ```dockerfile
   RUN useradd -m -u 1000 appuser
   USER appuser
   ```
3. **Scan for vulnerabilities:**
   ```bash
   docker scan scientific-agent
   ```
4. **Limit resources:**
   ```yaml
   deploy:
     resources:
       limits:
         cpus: '1'
         memory: 2G
   ```

## Monitoring

### View container stats
```bash
docker stats scientific-agent
```

### Check health
```bash
docker inspect --format='{{.State.Health.Status}}' scientific-agent
```

## Next Steps

- Set up reverse proxy (nginx/traefik) for production
- Add SSL/TLS certificates
- Set up monitoring and logging
- Configure auto-scaling based on load

