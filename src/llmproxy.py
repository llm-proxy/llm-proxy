import os
from models.openai import OpenAI
from models.vertexai import VertexAI
from dotenv import load_dotenv
import logging

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
project_id = os.getenv("GOOGLE_PROJECT_ID")


# Keep each function separate now for the sake a independently testing
def get_completion(prompt: str) -> str:
    # Using class allows us to not worry about passing in params every time we call a function
    openai = OpenAI(prompt=prompt, api_key=openai_api_key)
    res = openai.get_completion()

    if res.err:
        logging.error(f"OpenAI ErrorType: {res.err} \nError Message:{res.message}")

    return res.payload


def get_completion_vertexai(prompt: str) -> str:
    vertexai = VertexAI(prompt=prompt, project_id=project_id)
    res = vertexai.get_completion()

    if res.err:
        logging.error(f"Vertexai ErrorType: {res.err} \nError Message:{res.message}")

    return res.payload
