import os
from src.models import openai,vertex_ai


def getCompletion(prompt: str) -> str:
    openai_res = openai.get_open_ai_completion(prompt)

    if openai_res.err:
        return openai_res.message

    return openai_res.payload

def getVertexCompletion(prompt: str) -> str:
    vertex_res = vertex_ai.getAnswer(prompt)
    return vertex_res
