from flask import Flask, request, jsonify
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import os
import time
import uuid
from transformers import LlamaForCausalLM, LlamaTokenizer

app = Flask(__name__)
app.secret_key = "123456789"

# Define the chatbot's personality and introduction with explicit instructions
template = """
You are Nao, a friendly and engaging humanoid robot from Softbank Robotics. Start the conversation by introducing yourself with enthusiasm and warmth. Make sure to convey a sense of curiosity and willingness to assist.

Here is the conversation history: {context}

Question: {question}

Answer:
"""

CONTEXT_DIR = "contexts"
if not os.path.exists(CONTEXT_DIR):
    os.makedirs(CONTEXT_DIR)

# Store the time of the last message to handle inactivity
last_message_time = time.time()
current_context_file = f"{CONTEXT_DIR}/{str(uuid.uuid4())}.txt"

# Function to load context from file
def load_context(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            return file.read()
    return ""

# Function to save context to file
def save_context(context, file_path):
    with open(file_path, "w") as file:
        file.write(context)

# Function to dynamically generate a new context file
def new_context_file():
    return f"{CONTEXT_DIR}/{str(uuid.uuid4())}.txt"

# Function to dynamically load the latest checkpoint or final model
def load_latest_model():
    output_dir = './llama_finetune_checkpoints'
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith('checkpoint')]
    
    if checkpoints:
        # Get the latest checkpoint dynamically
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[-1]))
        latest_checkpoint_path = os.path.join(output_dir, latest_checkpoint)
        model = LlamaForCausalLM.from_pretrained(latest_checkpoint_path)
        tokenizer = LlamaTokenizer.from_pretrained(latest_checkpoint_path)
    else:
        # Fallback to final model if no checkpoints are found
        final_model_path = os.path.join(output_dir, 'final_model')
        model = LlamaForCausalLM.from_pretrained(final_model_path)
        tokenizer = LlamaTokenizer.from_pretrained(final_model_path)
    
    return model, tokenizer

# Initialize the model and tokenizer dynamically
model, tokenizer = load_latest_model()

@app.route('/chat', methods=['POST'])
def chat():
    global last_message_time, current_context_file
    data = request.json
    user_input = data.get('message')
    
    if user_input is None:
        return jsonify({"error": "No message provided"}), 400

    # Check for inactivity of 1 minute
    if time.time() - last_message_time > 60:
        # Save current context and start a new one
        current_context_file = new_context_file()

    context = load_context(current_context_file)
    
    # If user ends the conversation
    if "goodbye" in user_input.lower() or user_input.lower() in ["exit", "bye", "quit"]:
        response_text = "Nao: Goodbye! Have a great day!"
        save_context(context, current_context_file)  # Save current context
        current_context_file = new_context_file()  # Start a new context
        return jsonify({"response": response_text})
    
    # Get the response from the model (using the tokenizer for the input)
    input_ids = tokenizer(user_input, return_tensors='pt').input_ids
    result = model.generate(input_ids)
    response_text = tokenizer.decode(result[0], skip_special_tokens=True)
    
    # Update context
    context += f"\nUser: {user_input}\nNao: {response_text}"
    save_context(context, current_context_file)
    
    # Update last message time
    last_message_time = time.time()
    
    return jsonify({"response": response_text})

if __name__ == "__main__":
    app.run(debug=True)
