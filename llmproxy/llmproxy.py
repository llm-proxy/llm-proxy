import os
import yaml
import importlib
from llmproxy.models.cohere import Cohere
from llmproxy.utils.enums import BaseEnum
from typing import Any, Dict
from llmproxy.utils.log import logger
from llmproxy.utils.sorting import MinHeap

from dotenv import load_dotenv


class RouteType(str, BaseEnum):
    COST = "cost"
    CATEGORY = "category"


def _get_settings_from_yml(
    path_to_yml: str = "",
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
                    if model not in available_models[model_name]["models"]:
                        raise Exception(f"{model} is not available")

                    # Common params among all models
                    common_parameters = {
                        "max_output_tokens": provider["max_output_tokens"],
                        "temperature": provider["temperature"],
                        "model": model,
                    }

                    # Different setup for vertexai
                    if model_name == "vertexai":
                        common_parameters["project_id"] = os.getenv(
                            provider["project_id_var"]
                        )
                    else:
                        common_parameters["api_key"] = os.getenv(
                            provider["api_key_var"]
                        )

                    model_instance = available_models[model_name]["class"](
                        **common_parameters
                    )
                    user_models[model] = model_instance

        return user_models
    except Exception as e:
        raise e


class LLMProxy:
    def __init__(
        self,
        path_to_configuration: str = "api_configuration.yml",
        path_to_env_vars: str = ".env",
    ) -> None:
        self.user_models = {}
        self.route_type = "cost"

        load_dotenv(path_to_env_vars)

        # Read YML and see which models the user wants
        settings = _get_settings_from_yml(path_to_yml=path_to_configuration)
        # Setup available models
        available_models = _setup_available_models(settings=settings)

        self.user_models = _setup_user_models(
            settings=settings, available_models=available_models
        )

    # TODO: ROUTE TO ONLY AVAILABLE MODELS (check with adrian about this)
    # Do you want the model to route to the first available model
    # or just send back an error?
    def route(
        self, route_type: RouteType = RouteType.COST.value, prompt: str = ""
    ) -> str:
        if route_type not in RouteType:
            return "Sorry routing option available"

        if (route_type or self.route) == "cost":
            return self._cost_route(prompt=prompt)

        elif (route_type or self.route) == "category":
            print(self.user_models)
            return ""
            # Category routing
            pass

    def _cost_route(self, prompt: str):
        min_heap = MinHeap()
        for model_name, instance in self.user_models.items():
            logger.info(msg="========Start Cost Estimation===========")
            cost = instance.get_estimated_max_cost(prompt=prompt)
            logger.info(msg="========End Cost Estimation===========\n")

            item = {"name": model_name, "cost": cost, "instance": instance}
            min_heap.push(cost, item)

        completion_res = None
        while not completion_res:
            # Iterate through heap until there are no more options
            min_val_instance = min_heap.pop_min()
            if not min_val_instance:
                break

            instance_data = min_val_instance["data"]
            logger.info(f"Making request to model: {instance_data['name']}\n")
            logger.info("ROUTING...\n")

            # Attempt to make request to model
            try:
                # TODO: REMOVE COMPLETION RESPONSE TO SIMPLE raise exceptions to CLEAN UP CODE
                output = instance_data["instance"].get_completion(prompt=prompt)
                if output.payload and not output.err:
                    completion_res = output
                    logger.info("ROUTING COMPLETE! Call to model successful!\n")
                else:
                    logger.info("Request to model failed!\n")
                    logger.info(
                        f"Error when making request to model: '{output.message}'\n"
                    )
            except Exception as e:
                logger.info("Request to model failed!\n")
                logger.info(f"Error when making request to model: {e}\n")

        # If there is no completion_res raise exception
        if not completion_res:
            raise Exception(
                "Requests to all models failed! Please check your configuration!"
            )

        return completion_res
