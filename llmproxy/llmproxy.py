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
from llmproxy.utils.log import logger

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
        self.user_models = {}
        self.route_type = "cost"
        # Read YML and see which models the user wants
        settings = _get_settings_from_yml()
        # Setup available models
        available_models = _setup_available_models(settings=settings)
        self.user_models = _setup_user_models(
            settings=settings, available_models=available_models
        )

    def route(
        self, route_type: RouteType = RouteType.COST.value, prompt: str = ""
    ) -> str:
        if route_type not in RouteType:
            return "Sorry routing option available"

        chosen_model = None

        if route_type == "cost":
            output_model = {"name": "", "cost": float("inf"), "instance": None}

            for model_name, instance in self.user_models.items():
                logger.info(msg="========Start Cost Estimation===========")
                cost = instance.get_estimated_max_cost(prompt=prompt)
                logger.info(msg="========End Cost Estimation===========\n")

                if cost < output_model["cost"]:
                    output_model.update(
                        {"name": model_name, "cost": cost, "instance": instance}
                    )

            logger.info(f"Final model chosen: {output_model['name']}\n")
            chosen_model = output_model["instance"]

        elif route_type == "category":
            # Category routing
            pass

        logger.info(f"ROUTING...\n")
        completion_res = chosen_model.get_completion(prompt=prompt)
        logger.info(f"ROUTING COMPLETE")

        return completion_res


openai_api_key = os.getenv("OPENAI_API_KEY")
mistral_api_key = os.getenv("MISTRAL_API_KEY")
llama2_api_key = os.getenv("LLAMA2_API_KEY")
cohere_api_key = os.getenv("COHERE_API_KEY")
vertexai_project_id = os.getenv("GOOGLE_PROJECT_ID")

# Test Min Costs
prompt = "I am a man, not a man, but not a man, that is an apple, or a banana!"


# Test min cost example
def min_cost_cohere_example():
    cohere = Cohere(
        prompt=prompt,
        api_key=cohere_api_key,
        model="command",
        temperature=0,
    )
    return cohere.get_estimated_max_cost()


# Test completion example
def get_completion_cohere_example(prompt: str) -> str:
    cohere = Cohere(prompt=prompt, api_key=cohere_api_key)

    res = cohere.get_completion()
    if res.err:
        return res.message
    return res.payload
