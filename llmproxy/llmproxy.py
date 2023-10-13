import os

# from llmproxy.models import OpenAI
from llmproxy.models.openai import OpenAI
from llmproxy.models.cohere import Cohere
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
cohere_api_key = os.getenv("COHERE_API_KEY")

def get_completion(prompt: str) -> str:
    # Using class allows us to not worry about passing in params every time we call a function
    openai = OpenAI(prompt=prompt, api_key=openai_api_key)

    res = openai.get_completion()

    if res.err:
        return res.message

    return res.payload

def get_completion_cohere(prompt:str) -> str:
    # Using class allows us to not worry about passing in params every time we call a function
    cohere = Cohere(message=prompt, api_key=cohere_api_key)

    res = cohere.get_completion()

    if res.err:
        return res.message

    return res.payload