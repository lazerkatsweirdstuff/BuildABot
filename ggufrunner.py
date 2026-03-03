# app.py
from flask import Flask, request, jsonify, Response, render_template
import json
import os
import time
from flask_cors import CORS
from threading import Lock
import argparse
os.environ['TERM'] = 'dumb'  # Disable ANSI colors
# Try to import llama-cpp-python, provide instructions if not available
try:
    from llama_cpp import Llama
except ImportError:
    print("llama-cpp-python is not installed. Please install it with:")
    print("pip install llama-cpp-python")
    Llama = None

app = Flask(__name__)
app.lock = Lock()
app.llm = None

def load_model(model_path, n_ctx=2048, n_threads=4):
    """Load the GGUF model"""
    if Llama is None:
        raise ImportError("llama-cpp-python is not installed")
    
    print(f"Loading model from {model_path}...")
    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        verbose=False
    )
    print("Model loaded successfully!")
    return llm

@app.route('/')
def index():
    """Serve the chat interface"""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests and stream the response"""
    data = request.get_json()
    message = data.get('message', '')
    history = data.get('history', [])
    print(message)
    if not message:
        return jsonify({'error': 'No message provided'}), 400
    
    if app.llm is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    # Format the conversation history for the model
    formatted_history = format_history(history)
    full_prompt = formatted_history + f"User: {message}\nAssistant:"
    
    # Generate response with streaming
    def generate():
        with app.lock:  # Ensure only one request uses the model at a time
            stream = app.llm(
                full_prompt,
                max_tokens=256,
                stop=["User:", "###"],
                stream=True,
                temperature=0.7,
                top_p=0.9,
            )
            
            for output in stream:
                text = output['choices'][0]['text']
                yield f"data: {json.dumps({'text': text})}\n\n"
        
        yield "data: [DONE]\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

def format_history(history):
    """Format the chat history for the model"""
    formatted = ""
    for msg in history:
        if msg['role'] == 'user':
            formatted += f"User: {msg['content']}\n"
        else:
            formatted += f"Assistant: {msg['content']}\n"
    return formatted

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GGUF Chat Server')
    parser.add_argument('--model', type=str, required=True, help='Path to the GGUF model file')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to run the server on')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--n-ctx', type=int, default=2048, help='Context size for the model')
    parser.add_argument('--n-threads', type=int, default=4, help='Number of threads to use')
    
    args = parser.parse_args()
    
    # Load the model
    if Llama is not None:
        app.llm = load_model(args.model, args.n_ctx, args.n_threads)
    else:
        print("Running in demo mode without model. Install llama-cpp-python to use actual models.")
    
    # Run the Flask app
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)
    app = Flask(__name__)
    CORS(app)  # Add this line
