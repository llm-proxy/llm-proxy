import os
import yaml
import importlib
from llmproxy.models.cohere import Cohere
from llmproxy.models.llama2 import Llama2
from llmproxy.models.mistral import Mistral
from llmproxy.models.openai import OpenAI
from llmproxy.models.vertexai import VertexAI
from llmproxy.utils.enums import BaseEnum
from typing import Any, Dict

from dotenv import load_dotenv

load_dotenv()


class RouteType(str, BaseEnum):
    COST = "cost"
    CATEGORY = "category"


def _get_settings_from_yml(
    path_to_yml="api_configuration.yml",
) -> Dict[str, Any]:
    """Returns all of the settings in the api_configuration.yml file"""
    try:
        with open(path_to_yml, "r") as file:
            result = yaml.safe_load(file)
            return result
    except (FileNotFoundError, yaml.YAMLError) as e:
        raise e


def _setup_available_models(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Returns classname with list of available_models for provider"""
    try:
        available_models = {}
        # Loop through each "provider": provide means file name of model
        for provider in settings["available_models"]:
            key = provider["name"].lower()
            import_path = provider["class"]

            # Loop through and aggreate all of the variations of "models" of each provider
            provider_models = set()
            for model in provider.get("models"):
                provider_models.add(model["name"])

            module_name, class_name = import_path.rsplit(".", 1)

            module = importlib.import_module(module_name)
            model_class = getattr(module, class_name)

            # return dict with class path and models set, with all of the variations/models of that provider
            available_models[key] = {"class": model_class, "models": provider_models}

        return available_models
    except Exception as e:
        raise e


def _setup_user_models(available_models={}, settings={}) -> Dict[str, object]:
    """Setup all available models and return dict of {name: instance_of_model}"""
    try:
        user_models = {}
        # Compare user models with available_models
        for provider in settings["user_settings"]:
            model_name = provider["model"].lower().strip()
            # Check if user model in available models
            if model_name in available_models:
                # If the user providers NO variations then raise error
                if "models" not in provider or provider["models"] is None:
                    raise Exception("No models provided in user_settings")

                # Loop through and set up instance of model
                for model in provider["models"]:
                    # Different setup for vertexai
                    if model_name == "vertexai":
                        if model in available_models[model_name]["models"]:
                            model_instance = available_models[model_name]["class"](
                                project_id=os.getenv("GOOGLE_PROJECT_ID"),
                                max_output_tokens=provider["max_output_tokens"],
                                temperature=provider["temperature"],
                                model=model,
                            )
                            user_models[model] = model_instance
                        else:
                            raise Exception(model + " is not a valid model")
                    else:
                        # Same setup for others
                        if model in available_models[model_name]["models"]:
                            model_instance = available_models[model_name]["class"](
                                api_key=os.getenv(provider["api_key_var"]),
                                max_output_tokens=provider["max_output_tokens"],
                                temperature=provider["temperature"],
                                model=model,
                            )
                            user_models[model] = model_instance
                        else:
                            raise Exception(model + " is not a valid model")

        return user_models
    except Exception as e:
        raise e


class LLMProxy:
    def __init__(self) -> None:
        # ... Read YML and see which models the user wants
        self.user_models = {}
        self.route_type = "cost"

        settings = _get_settings_from_yml()

        # Setup available models
        available_models = _setup_available_models(settings=settings)

        self.user_models = _setup_user_models(
            settings=settings, available_models=available_models
        )

    def route(self, route_type: RouteType = RouteType.COST.value) -> str:
        if route_type not in RouteType:
            return "Sorry routing option available"
        # for model, instance in self.user_models.items():
        # op = instance.get_completion(prompt="HELLLOO, what is 1+1?")
        # print(f"{model}: {op}")
        if route_type == "cost":
            return ""
        elif route_type == "category":
            # Category routing
            return ""


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
        prompt=prompt, api_key=llama2_api_key, model="Llama-2-7b-chat-hf", temperature=0
    )

    return llama2.get_estimated_max_cost()


def min_cost_mistral():
    mistral = Mistral(
        prompt=prompt, api_key=mistral_api_key, model="Mistral-7B-v0.1", temperature=0
    )

    return mistral.get_estimated_max_cost()


def min_cost_vertexai():
    vertexai = VertexAI(
        prompt=prompt,
        project_id=vertexai_project_id,
        model="text-bison@001",
        temperature=0,
    )

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
