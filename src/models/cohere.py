import cohere
import os

from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv('API_KEY')

co = cohere.Client(api_key)

def cohere_ai_completion(prompt):
    response = co.chat(
    # chat_history is not a needed parameter
    chat_history=[
        {"role": "USER", "message": "Who discovered gravity?"},
        {"role": "CHATBOT", "message": "The man who is widely credited with discovering gravity is Sir Isaac Newton"}
    ],
    message=prompt,
    connectors=[{"id": "web-search"}] # perform web search before answering the question
    )
    return response.text

prompt = "How many years ago was the universe created?"
print(cohere_ai_completion(prompt))
