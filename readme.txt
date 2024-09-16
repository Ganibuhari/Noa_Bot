ZA-3201: Assignment 2 Part 1

TASK 3: DEVELOP A BASIC CONVERSATION BOT ON A PC BASED ON SIMPLE RULES.

FIRST STEP: Finding a LLM

In this case, we decided to use LLAMA, a free large language model (LLM).

OpenAI's models require a paid subscription.

so head over to this website https://ollama.com/ (OLLAMA)

Ollama is a platform that offers a range of large language models (LLMs) specifically designed for local, on-device use.

Since it is local used the LLM doesnt require any internet connection to be able to get responses from LLM.

Ollama supports several LLMs, including LLAMA which we will be using as well as PHI3,Mistral and Gemma 2.

Download the OLLAMA installer and run the .exe

SECOND STEP: Downloading the LLM model

The requirement is atleast 4.7GB,this LLM contains 7 Billion parameters.

Once it is done,open terminal on your hardware

and type in this command "ollama run llama3.1"

Above command will install the LLAMA3.1 model on your hardware.

THIRD STEP: Activating and using LLM model

Open terminal on your hardware.

Type in "Ollama run llama3.1"

This will start the LLM model and you can start asking the LLM model questions.

FOURTH STEP:Setting up virtual environment

In this case we are using python version 3.9.13

Go to visual studio and open a new terminal

Type "python3.9 -m venv <myenvname>" to create a virtual environment

To activate the environment "Scripts\activate" to deactivate type "deactivate"

FIFTH STEP:Installing the required packages for LLM model

Activate your virtual environment and type "pip install -r requirements.txt"

The require packages should automatically be installed.

SXTH STEP: Sign up for POSTMAN

Postman is a popular tool used for testing and interacting with APIs (Application Programming Interfaces). It provides a user-friendly interface for making HTTP requests and inspecting responses.

Postman makes it easy to send HTTP requests (GET, POST, PUT, DELETE, etc.) without writing code.

Head over to https://www.postman.com/ and sign up

Once sign up create and name your workspace.

Create a new request

At the request set method to POST

After that the url will be this "http://127.0.0.1:5000/chat"

Head over to "Body" and the "raw" column

Here we will write a code to send messages to the LLM

Copy and Paste the code below

{
   "messages": "Hi Nao"
}


With this we are able to send request and get responses from the LLama

SEVENTH STEP:Running the scripts

In Visual Studio code,activate your environment 

Open a new terminal and type "python main.py"

After that you are all set with a NAO CHATBOT

Head over to POSTMAN and type in your request.






 