"""
Ollama chat server

pip install langchain==0.3.1 langchain-community==0.3.1 ollama==0.3.3 gradio==4.42.0
"""
import gradio as gr
from langchain.llms import Ollama
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler
import asyncio

# Available Ollama models
MODELS = ["phi3:latest", "llama3.1:latest", "mistral-nemo:latest",]
TEMPERATURE = 0.0

# Custom callback handler for Gradio streaming
class GradioStreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.generated_text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.generated_text += token

# Initialize the conversation chain with memory
def create_chain(model_name):
    llm = Ollama(model=model_name,
                 temperature=TEMPERATURE,
        )
    memory = ConversationBufferMemory()
    return ConversationChain(llm=llm, memory=memory)

# Global variable to store the current conversation chain
current_chain = create_chain(MODELS[0])

# Function to generate a response
async def generate_response(message, history, model):
    global current_chain
    if model != current_chain.llm.model:
        current_chain = create_chain(model)
    
    streaming_callback = GradioStreamingCallbackHandler()
    
    async def async_predict():
        return await asyncio.to_thread(
            current_chain.predict,
            input=message,
            callbacks=[streaming_callback]
        )

    task = asyncio.create_task(async_predict())

    while not task.done():
        yield streaming_callback.generated_text
        await asyncio.sleep(0.05)
    
    yield streaming_callback.generated_text

# Function to change the model
def change_model(model):
    global current_chain
    current_chain = create_chain(model)
    return f"Model changed to {model}"

# Create the Gradio interface
with gr.Blocks() as iface:
    gr.Markdown('<h1 style="text-align: center;">🦜 OllamaChat 🤖</h1>\n ')
    
    with gr.Row():
        model_dropdown = gr.Dropdown(choices=MODELS, value=MODELS[0], label="Select Model")
        change_model_btn = gr.Button("Change Model")

    change_model_btn.click(change_model, inputs=[model_dropdown], outputs=[gr.Textbox()])

    chatbot = gr.ChatInterface(
        generate_response,
        additional_inputs=[model_dropdown],
        title="",
    )

# Launch the app
if __name__ == "__main__":
    iface.launch()
