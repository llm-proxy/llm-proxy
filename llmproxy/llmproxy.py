import os

from llmproxy.models.openai import OpenAI
from llmproxy.models.mistral import Mistral
from llmproxy.models.cohereai import Cohere

from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
mistral_api_key = os.getenv("MISTRAL_API")
cohere_api_key = os.getenv("COHERE_API_KEY")


def get_completion(prompt: str) -> str:
    # Using class allows us to not worry about passing in params every time we call a function
    openai = OpenAI(prompt=prompt, api_key=openai_api_key)

    res = openai.get_completion()

    if res.err:
        return res.message

    return res.payload


def get_completion_mistral(prompt: str) -> str:
    mistral = Mistral(prompt=prompt, api_key=mistral_api_key)

    res = mistral.get_completion()


def get_completion_cohere(prompt: str) -> str:
    # Using class allows us to not worry about passing in params every time we call a function
    cohere = Cohere(message=prompt, api_key=cohere_api_key)

    res = cohere.get_completion()

    if res.err:
        return res.message

    return res.payload
