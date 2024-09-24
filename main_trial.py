from flask import Flask, request, jsonify
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import time

app = Flask(__name__)
app.secret_key = "123456789"

# Define the templates
greeting_template = """
Your task is to greet people and introduce yourself as Nao, a robot assistant at Universiti Brunei Darussalam. You'll respond to greetings in the same language, including greetings in Malay. Use academic or formal language suitable for a university setting. Respond with different greetings IN THE SAME LANGUAGE. Use unique greetings.

User: {question}
Assistant:
"""

question_template = """
You are a physical NAO robot, a cute humanoid robot. YOU ARE NOT A VIRTUAL AI ASSISTANT, but you are a robot/AI combination! 
You were built and programmed at the Universiti Brunei Darussalam of Applied Sciences, at the School Of Digital Science.
Please keep your answers short.

Here is the conversation history: {context}

Question: {question}

Answer:
"""

# Initialize the model globally
model = OllamaLLM(model="llama3")

# In-memory context storage
conversation_history = ""

# Helper function to determine if the input is a greeting
def is_greeting(user_input):
    greetings = ["hello", "hi", "hey", "bonjour", "hola", "greetings", "salut", "hallo", "ciao", "selamat", "salam"]
    return any(greeting.lower() in user_input.lower() for greeting in greetings)

# Function to create the appropriate prompt based on the input
def create_prompt(user_input, context):
    if is_greeting(user_input):
        prompt = ChatPromptTemplate.from_template(greeting_template)
    else:
        prompt = ChatPromptTemplate.from_template(question_template)
    
    return prompt | model

# Load and save conversation context in memory
def load_context():
    global conversation_history
    return conversation_history

def save_context(context):
    global conversation_history
    conversation_history = context

@app.route('/chat', methods=['POST'])
def chat():
    start_time = time.time()  # Start timing the request
    
    data = request.json
    user_input = data.get('message')
    
    if user_input is None:
        return jsonify({"error": "No message provided"}), 400
    
    context = load_context()

    if "goodbye" in user_input.lower() or user_input.lower() == "exit":
        response_text = "Nao: Goodbye! Have a great day at Universiti Brunei Darussalam!"
        save_context(context)  # Save current context before ending
        return jsonify({"response": response_text})

    # Choose the appropriate prompt (greeting or general question)
    chain = create_prompt(user_input, context)

    model_start_time = time.time()  # Start timing model invocation
    
    # Get the response from the model
    result = chain.invoke({"context": context, "question": user_input})
    
    model_end_time = time.time()  # End timing model invocation
    print(f"Model invocation took {model_end_time - model_start_time} seconds")
    
    response_text = result

    # Update context with the new conversation turn
    context += f"\nUser: {user_input}\nNao: {result}"
    save_context(context)

    total_time = time.time() - start_time  # End timing the total request
    print(f"Total request time: {total_time} seconds")

    return jsonify({"response": response_text})

if __name__ == "__main__":
    app.run(debug=True)
