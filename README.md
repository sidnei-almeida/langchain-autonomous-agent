---
title: Scientific Research Agent
emoji: 🔬
colorFrom: purple
colorTo: blue
sdk: docker
app_file: app.py
pinned: false
license: mit
---

# 🔬 Scientific Research Agent

An autonomous AI agent specialized in scientific research that uses multiple tools to provide accurate, evidence-based answers. Built with LangChain, Groq (Llama 3), and scientific research tools.

## 🛠️ Available Tools

The agent has access to the following scientific tools:

- **🌐 Web Search (DuckDuckGo)**: For up-to-date information, news, and recent events
- **📚 Wikipedia**: For detailed encyclopedic information and general concepts
- **🔬 ArXiv**: To search and retrieve scientific articles, academic papers, and scientific literature
- **🧮 Scientific Calculator**: For complex mathematical calculations including trigonometric, logarithmic, and exponential functions

## 🚀 Installation

1. Clone the repository:
```bash
git clone <repo-url>
cd langchain-autonomous-agent
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure your Groq API key:
```bash
# Edit the .env file and add your key:
GROQ_API_KEY="your_key_here"
```

## 📖 Usage

### CLI Mode (Interactive)
```bash
python agent.py
```

### CLI Mode (Command Line Argument)
```bash
python agent.py "What are the latest advances in machine learning?"
```

### Web Interface (Gradio)
```bash
python app.py
```

Then open your browser to `http://localhost:7860`

### Docker Deployment

#### Using Docker Compose (Recommended)
```bash
# Make sure your .env file has GROQ_API_KEY set
docker-compose up --build
```

The app will be available at `http://localhost:7860`

#### Using Docker directly
```bash
# Build the image
docker build -t scientific-agent .

# Run the container
docker run -p 7860:7860 -e GROQ_API_KEY=your_key_here scientific-agent
```

#### Stop the container
```bash
# If using docker-compose
docker-compose down

# If using docker directly
docker stop <container_id>
```

## 🌐 Deployment Options

### Option 1: Hugging Face Spaces (No Docker needed)

This project is ready to deploy on Hugging Face Spaces without Docker!

**Steps to Deploy:**

1. **Create a new Space** on [Hugging Face Spaces](https://huggingface.co/spaces)
   - Choose **Gradio** as the SDK
   - Select **Python** as the runtime

2. **Upload your files**:
   - `app.py` (main application)
   - `agent.py` (agent logic)
   - `requirements.txt` (dependencies)

3. **Set up secrets**:
   - Go to your Space settings
   - Add a secret named `GROQ_API_KEY` with your Groq API key

4. **Deploy**:
   - The Space will automatically build and deploy
   - Your agent will be available at `https://huggingface.co/spaces/your-username/your-space-name`

**Note:** Hugging Face Spaces handles containerization automatically, so you don't need Docker for this option.

### Option 2: Docker Deployment

For deploying on other platforms (AWS, Google Cloud, Azure, etc.) or local containerized environments:

**Using Docker Compose:**
```bash
# Set your API key in .env file
echo "GROQ_API_KEY=your_key_here" > .env

# Build and run
docker-compose up --build
```

**Using Docker directly:**
```bash
# Build image
docker build -t scientific-agent .

# Run container
docker run -p 7860:7860 -e GROQ_API_KEY=your_key_here scientific-agent
```

**Benefits of Docker:**
- Consistent environment across different platforms
- Easy deployment to cloud services
- Isolated dependencies
- Production-ready configuration

### Environment Variables

The agent automatically reads the `GROQ_API_KEY` from:
- Environment variables (Docker, Hugging Face Spaces secrets)
- `.env` file (local development)

## 💡 Example Questions

- "Search for recent articles about quantum computing on ArXiv"
- "Explain the concept of entropy using Wikipedia and calculate some examples"
- "What are the latest discoveries about CRISPR?"
- "Calculate sin(pi/2) + cos(0) + sqrt(16)"
- "Find recent papers on machine learning from ArXiv"

## 🔧 Technologies

- **LangChain**: Framework for building AI agents
- **Groq**: LLM API with Llama 3.3 70B model
- **Gradio**: Web interface framework
- **DuckDuckGo Search**: Web search
- **Wikipedia API**: Encyclopedia access
- **ArXiv API**: Scientific articles access
- **Python**: Programming language

## 📝 Project Structure

```
langchain-autonomous-agent/
├── agent.py          # Main agent code
├── app.py            # Gradio web interface (for Hugging Face Spaces)
├── requirements.txt  # Project dependencies
├── Dockerfile        # Docker image configuration
├── docker-compose.yml # Docker Compose configuration
├── .dockerignore     # Files to exclude from Docker build
├── .env             # Environment variables (not versioned)
└── README.md        # This file
```

## 🎯 Features

- ✅ Access to multiple scientific information sources
- ✅ Automatic search in ArXiv scientific articles
- ✅ Complex mathematical calculations
- ✅ Interactive and user-friendly interface
- ✅ Evidence-based answers
- ✅ Source citation when relevant
- ✅ Ready for Hugging Face Spaces deployment

## 📄 License

This project is open source and available for educational and research use.
