import os
import yaml
import importlib
from llmproxy.utils.enums import BaseEnum
from typing import Any, Dict

from dotenv import load_dotenv
import os

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
mistral_api_key = os.getenv("MISTRAL_API_KEY")
llama2_api_key = os.getenv("LLAMA2_API_KEY")
cohere_api_key = os.getenv("COHERE_API_KEY")
vertexai_project_id = os.getenv("GOOGLE_PROJECT_ID")


class RouteType(str, BaseEnum):
    COST = "cost"
    CATEGORY = "category"


def _get_settings_from_yml(
    path_to_yml="api_configuration.yml",
) -> Dict[str, Any]:
    try:
        with open(path_to_yml, "r") as file:
            result = yaml.safe_load(file)
            print((result))
            return result
    except (FileNotFoundError, yaml.YAMLError) as e:
        raise e


def _setup_available_models(settings={}) -> Dict[str, Any]:
    try:
        available_models = {}
        for provider in settings["available_models"]:
            key = provider["name"].lower()
            import_path = provider["class"]

            provider_models = set()
            for model in provider.get("models", []):
                provider_models.add(model["name"])

            print(provider_models)

            module_name, class_name = import_path.rsplit(".", 1)

            module = importlib.import_module(module_name)
            model_class = getattr(module, class_name)

            available_models[key] = {"class": model_class, "models": provider_models}
        return available_models
    except Exception as e:
        raise e


def _setup_user_models(available_models={}, settings={}) -> Dict[str, object]:
    # Setup all user models
    try:
        user_models = {}
        for provider in settings["user_settings"]:
            model_name = provider["model"].lower().strip()
            if model_name in available_models:
                # Different setup for vertexai
                if "models" not in provider or provider["models"] is None:
                    raise Exception("No models provided in user_settings")
                for model in provider["models"]:
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
        print(self.user_models)
        if route_type == "cost":
            # Cost routing
            return ""
        elif route_type == "category":
            # Category routing
            return ""
        # for name, instance in user_models.items():
        #     res = instance.get_completion(prompt="What's 1 + 1 + 2 + 2?")
        #         if res.err:
        #             print(name, res.message)
        #         else:
        #             print(name, res.payload)
