# 🔬 Scientific Research Agent - Hugging Face Spaces

This is a ready-to-deploy Gradio application for Hugging Face Spaces.

## Quick Deploy Guide

1. **Create a new Space**:
   - Go to [Hugging Face Spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - Name: `scientific-research-agent` (or your preferred name)
   - SDK: **Gradio**
   - Hardware: **CPU Basic** (free tier works fine)

2. **Upload Files**:
   - Upload `app.py`
   - Upload `agent.py`
   - Upload `requirements.txt`

3. **Set Secret**:
   - Go to Settings → Secrets
   - Add a new secret:
     - Key: `GROQ_API_KEY`
     - Value: Your Groq API key (get it from [console.groq.com](https://console.groq.com))

4. **Deploy**:
   - The Space will automatically build
   - Wait for the build to complete (usually 2-5 minutes)
   - Your app will be live!

## File Structure for HF Spaces

```
your-space/
├── app.py            # Main Gradio app (required)
├── agent.py          # Agent logic
├── requirements.txt  # Dependencies
└── README.md         # This file (optional but recommended)
```

## Environment Variables

The app uses the `GROQ_API_KEY` environment variable, which you set through Hugging Face Spaces secrets. The code automatically reads it, so no `.env` file is needed.

## Testing Locally

Before deploying, you can test locally:

```bash
# Set your API key
export GROQ_API_KEY="your_key_here"

# Run the app
python app.py
```

Then visit `http://localhost:7860` in your browser.

## Troubleshooting

- **Build fails**: Check that all dependencies are in `requirements.txt`
- **API key error**: Verify the secret is set correctly in Space settings
- **Import errors**: Make sure `agent.py` is in the same directory as `app.py`

## Notes

- The agent initializes once at startup for better performance
- Responses may take 10-30 seconds depending on the complexity of the query
- The app uses Groq's free tier, which has rate limits

