import os

from llmproxy.models.openai import OpenAI
from llmproxy.models.mistral import Mistral
from llmproxy.models.llama2 import Llama2
from llmproxy.models.vertexai import VertexAI
from llmproxy.models.cohere import Cohere

from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
mistral_api_key = os.getenv("MISTRAL_API_KEY")
llama2_api_key = os.getenv("LLAMA2_API_KEY")
cohere_api_key = os.getenv("COHERE_API_KEY")
vertexai_project_id = os.getenv("GOOGLE_PROJECT_ID")


# Test Min Costs
prompt = "I am a man, not a man, but not a man, that is an apple, or a banana!"


def min_cost_openai():
    openai = OpenAI(
        prompt=prompt,
        api_key="",
        model="gpt-3.5-turbo-1106",
        temperature=0,
    )
    return openai.get_estimated_max_cost()


def min_cost_cohere():
    cohere = Cohere(
        prompt=prompt,
        api_key=cohere_api_key,
        model="command",
        temperature=0,
    )
    return cohere.get_estimated_max_cost()


def min_cost_llama2():
    llama2 = Llama2(
        prompt=prompt,
        api_key=llama2_api_key,
        model="Llama-2-7b-chat-hf",
        temperature=0)

    return llama2.get_estimated_max_cost()


def min_cost_mistral():
    mistral = Mistral(
        prompt=prompt,
        api_key=mistral_api_key,
        model="Mistral-7B-v0.1",
        temperature=0)

    return mistral.get_estimated_max_cost()


def min_cost_vertexai():
    vertexai = VertexAI(
        prompt=prompt,
        project_id=vertexai_project_id,
        model="text-bison@001",
        temperature=0)

    return vertexai.get_estimated_max_cost()


# Test completion
def get_completion_openai(prompt: str) -> str:
    # Using class allows us to not worry about passing in params every time we
    # call a function
    openai = OpenAI(prompt=prompt, api_key=openai_api_key)

    res = openai.get_completion()

    if res.err:
        return res.message

    return res.payload


def get_completion_mistral(prompt: str) -> str:
    mistral = Mistral(prompt=prompt, api_key=mistral_api_key)

    res = mistral.get_completion()

    if res.err:
        return res.message

    return res.payload


def get_completion_llama2(prompt: str) -> str:
    llama = Llama2(prompt=prompt, api_key=llama2_api_key)

    res = llama.get_completion()

    if res.err:
        return res.message
    return res.payload


def get_completion_cohere(prompt: str) -> str:
    cohere = Cohere(prompt=prompt, api_key=cohere_api_key)

    res = cohere.get_completion()
    if res.err:
        return res.message
    return res.payload


def get_completion_vertexai(prompt: str, location: str = "us-central1") -> str:
    vertexai = VertexAI(
        prompt=prompt, location=location, project_id=vertexai_project_id
    )

    res = vertexai.get_completion()

    if res.err:
        return res.message

    return res.payload
