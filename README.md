# ollama-chat-server
Gradio chat server using ollama. Good for offline usage!

## Instructions
1. Install Ollama - https://ollama.com/
2. Install requirements - `pip install -r requirements.txt`
4. Download ollama models locally according to local memory and desired performance - `ollama pull <model-name>`
5. Change the line code containing `MODELS` in the `chatserver.py` with the downloaded models
6. Start server - `python chatserver.py`
