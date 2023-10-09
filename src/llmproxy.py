import os
from models.openai import OpenAI
from models.mistral import Mistral
from dotenv import load_dotenv


load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
mistral_api_key = os.getenv('MISTRAL_API')


def getCompletion(prompt: str) -> str:
    # Using class allows us to not worry about passing in params every time we call a function
    openai = OpenAI(prompt=prompt, api_key=openai_api_key)

    if res.err:
        return res.message

    return res.payload

def textGenerate(prompt: str) -> str:
    mistral = Mistral(
        prompt = prompt,
        api_key = mistral_api_key,
    )

    return mistral.get_completion()
