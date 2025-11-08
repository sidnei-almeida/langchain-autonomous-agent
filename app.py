"""
Main application file for Hugging Face Spaces
This file imports and runs the FastAPI application
"""
from api import app
import uvicorn
import os

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
