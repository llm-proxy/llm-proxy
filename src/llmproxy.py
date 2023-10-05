import os
from models import openai


def getCompletion(prompt: str) -> str:
    openai_res = openai.get_open_ai_completion(prompt)

    if openai_res.err:
        return openai_res.message

    return openai_res.payload
