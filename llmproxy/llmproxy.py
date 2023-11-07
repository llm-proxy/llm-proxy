import os
import yaml

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

# mapping models to their respective completion function
completion_functions = {
    "OpenAI": get_completion,
    "Cohere": get_completion_cohere,
    "Llama2": get_completion_llama2,
    "Mistral": get_completion_mistral,
}


# use by user for prompting based on user_setting in the api_configuration.yml
def prompt(prompt: str) -> str:
    result = {}
    try:
        with open("api_configuration.yml", "r") as file:
            settings = yaml.safe_load(file)

        if settings is None or 'user_setting' not in settings:
            raise ValueError("Invalid or missing 'user_setting' in api_configuration.yaml")
        
        for setting in settings['user_setting']:
            model = setting.get('model')
            if model is not None and model in completion_functions:
                parameters = setting
                completion_function = completion_functions[model]
                result = completion_function(prompt, **parameters)
                return result
            else:
                result[model] = f"Model '{model}' not found or completion_function not defined."

    except (FileNotFoundError, yaml.YAMLError) as e:
        result['Error'] = f"An error occurred: {e}"

    return result


def get_completion_openai(prompt: str) -> str:
    # Using class allows us to not worry about passing in params every time we call a function
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
