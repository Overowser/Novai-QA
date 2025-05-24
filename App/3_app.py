from generator import generate_response
import gradio as gr
from sentence_transformers import SentenceTransformer
from logger_config import setup_logger

logger = setup_logger("app")
logger.info("Initializing SentenceTransformer model")
model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", device="cuda")
logger.info("Model initialized successfully")

def respond(message, history, novel_name, spoiler_threshold):
    """
    Function that gets the complete response directly
    """
    logger.info("Received input - Novel: %s, Spoiler Threshold: %s, Query: %s",
                novel_name, spoiler_threshold, message)
    try:
        response = generate_response(message, novel_name, model, spoiler_threshold)
        logger.info("Generated response: %s", response[:100] + "..." if len(response) > 100 else response)
        return response
    except Exception as e:
        logger.error("Error generating response: %s", str(e))
        return f"Sorry, I encountered an error: {str(e)}"

# Create the Gradio interface with ChatInterface for better UX
with gr.Blocks(title="Novel Assistant Chatbot", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Novel Assistant Chatbot")
    gr.Markdown("Ask questions about your favorite novels! The bot will avoid spoilers based on your current chapter.")
   
    with gr.Row():
        with gr.Column(scale=1):
            novel_name = gr.Textbox(
                label="Novel Name",
                placeholder="Enter the name of the novel...",
                value=""
            )
            spoiler_threshold = gr.Number(
                label="Current Chapter (optional)",
                value=None,
                minimum=0
            )
       
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="Chat",
                height=500,
                show_label=False,
                container=True,
                bubble_full_width=False
            )
           
            msg = gr.Textbox(
                label="Message",
                placeholder="Ask your question about the novel...",
                show_label=False,
                container=False
            )
           
            with gr.Row():
                submit = gr.Button("Send", variant="primary")
                clear = gr.Button("Clear Chat", variant="secondary")

    def user_message(message, history):
        """Add user message to chat history"""
        if not message.strip():  # Don't add empty messages
            return message, history
        return "", history + [[message, None]]

    def bot_response(history, novel_name, spoiler_threshold):
        """Generate bot response"""
        if not history or not history[-1][0]:  # Check if history exists and has user message
            logger.warning("No history or empty user message")
            return history
           
        user_message = history[-1][0]
        logger.info("Processing user message: %s", user_message)
       
        try:
            # Get the complete response
            response = respond(user_message, history, novel_name, spoiler_threshold)
            
            # Ensure response is not None or empty
            if not response:
                response = "I'm sorry, I couldn't generate a response. Please try again."
                logger.warning("Empty response generated")
            
            # Set the bot response
            history[-1][1] = response
            logger.info("Bot response set successfully")
            
        except Exception as e:
            logger.error("Error in bot_response: %s", str(e))
            history[-1][1] = f"Sorry, I encountered an error: {str(e)}"
        
        return history

    # Event handlers
    msg.submit(
        user_message,
        [msg, chatbot],
        [msg, chatbot],
        queue=False
    ).then(
        bot_response,
        [chatbot, novel_name, spoiler_threshold],
        [chatbot],  # Only return chatbot, not multiple outputs
        queue=False
    )
   
    submit.click(
        user_message,
        [msg, chatbot],
        [msg, chatbot],
        queue=False
    ).then(
        bot_response,
        [chatbot, novel_name, spoiler_threshold],
        [chatbot],  # Only return chatbot, not multiple outputs
        queue=False
    )
   
    clear.click(lambda: ([], ""), outputs=[chatbot, msg])

if __name__ == "__main__":
    logger.info("Launching Gradio demo")
    demo.launch()