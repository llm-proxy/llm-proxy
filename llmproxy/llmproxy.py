import os

from llmproxy.models.openai import OpenAI
from llmproxy.models.mistral import Mistral
from llmproxy.models.llama2 import Llama2

from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
mistral_api_key = os.getenv("MISTRAL_API_KEY")
llama2_api_key = os.getenv("LLAMA2_API_KEY")


def min_cost():
    openai = OpenAI(prompt="I am a man, not a man", api_key="", model="gpt-4", temp=0)
    print(openai.get_estimated_max_cost())


def get_completion(prompt: str) -> str:
    # Using class allows us to not worry about passing in params every time we call a function
    openai = OpenAI(prompt=prompt, api_key=openai_api_key)

    res = openai.get_completion()

    if res.err:
        return res.message

    return res.payload


def get_completion_mistral(prompt: str, model: str) -> str:
    mistral = Mistral(prompt=prompt, api_key=mistral_api_key, model=model)

    res = mistral.get_completion()

    if res.err:
        return res.message

    return res.payload


def get_completion_llama2(prompt: str, system_prompt: str, model: str) -> str:
    llama = Llama2(
        prompt=prompt, system_prompt=system_prompt, api_key=llama2_api_key, model=model
    )

    res = llama.get_completion()

    if res.err:
        return res.message
    return res.payload
