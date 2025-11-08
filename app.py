"""
Hugging Face Spaces App for Scientific Agent
This file creates a Gradio interface for the scientific agent.
"""
import os
import gradio as gr
from agent import create_scientific_agent
from langchain_core.messages import AIMessage

# Initialize the agent (will be created once at startup)
agent = None

def initialize_agent():
    """Initialize the agent on startup."""
    global agent
    if agent is None:
        try:
            agent = create_scientific_agent()
            return "✅ Agent initialized successfully!"
        except Exception as e:
            return f"❌ Error initializing agent: {str(e)}"
    return "✅ Agent already initialized!"

def query_agent(question: str, history: list) -> tuple:
    """
    Query the scientific agent with a question.
    
    Args:
        question: The user's question
        history: Chat history (list of [user_message, bot_message] pairs)
    
    Returns:
        Updated history with the new question and answer
    """
    global agent
    
    if not question or not question.strip():
        return history
    
    if agent is None:
        try:
            agent = create_scientific_agent()
        except Exception as e:
            error_msg = f"Error initializing agent: {str(e)}"
            history.append([question, error_msg])
            return history
    
    try:
        # Invoke the agent
        result = agent.invoke({"messages": [{"role": "user", "content": question}]})
        messages = result.get("messages", [])
        
        # Extract the final answer
        final_answer = next(
            (msg.content for msg in reversed(messages) if isinstance(msg, AIMessage)),
            None,
        )
        
        if final_answer:
            answer = final_answer
        else:
            answer = "I couldn't generate a proper answer. Please try rephrasing your question."
        
        # Append to history
        history.append([question, answer])
        
    except Exception as e:
        error_msg = f"Error processing your question: {str(e)}"
        history.append([question, error_msg])
    
    return history

# Initialize agent on startup
init_status = initialize_agent()
print(init_status)

# Create Gradio interface
with gr.Blocks(title="🔬 Scientific Research Agent", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🔬 Scientific Research Agent
        
        An autonomous AI agent specialized in scientific research with access to multiple tools:
        
        - 🌐 **Web Search** (DuckDuckGo): Up-to-date information, news, and recent events
        - 📚 **Wikipedia**: Detailed encyclopedic information and concepts
        - 🔬 **ArXiv**: Scientific articles, academic papers, and research literature
        - 🧮 **Scientific Calculator**: Complex mathematical and scientific calculations
        
        Ask any scientific question and the agent will use the appropriate tools to provide a comprehensive, evidence-based answer.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 💡 Example Questions")
            examples = gr.Examples(
                examples=[
                    "What are the latest advances in quantum computing according to ArXiv?",
                    "Explain the concept of entropy using Wikipedia and calculate some examples",
                    "What are the recent discoveries about CRISPR?",
                    "Calculate sin(pi/2) + cos(0) + sqrt(16)",
                    "Find recent papers on machine learning from ArXiv",
                ],
                inputs=None,
            )
        
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="Scientific Agent",
                height=500,
                show_copy_button=True,
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask your scientific question here...",
                    scale=4,
                    lines=2,
                )
                submit_btn = gr.Button("Ask 🔬", scale=1, variant="primary")
            
            clear_btn = gr.Button("Clear Chat", variant="secondary")
    
    # Event handlers
    msg.submit(query_agent, [msg, chatbot], [chatbot]).then(
        lambda: "", None, [msg]
    )
    submit_btn.click(query_agent, [msg, chatbot], [chatbot]).then(
        lambda: "", None, [msg]
    )
    clear_btn.click(lambda: [], None, [chatbot])
    
    gr.Markdown(
        """
        ---
        **Note**: Make sure to set your `GROQ_API_KEY` as a secret in Hugging Face Spaces settings.
        The agent uses Groq's Llama 3.3 70B model for reasoning and tool selection.
        """
    )

if __name__ == "__main__":
    # For Hugging Face Spaces, use share=False
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

