import os
from models import openai
from models.mistral import Mistral
from dotenv import load_dotenv

load_dotenv()

mistral_api_key = os.getenv('MISTRAL_API')

def getCompletion(prompt: str) -> str:
    openai_res = openai.get_open_ai_completion(prompt)

    if openai_res.err:
        return openai_res.message

    return openai_res.payload

def textGenerate(prompt: str) -> str:
    mistral = Mistral(
        prompt = prompt,
        api_key = mistral_api_key,
    )

    return mistral.get_completion()
