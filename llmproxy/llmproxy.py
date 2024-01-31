import importlib
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal

import yaml
from dotenv import load_dotenv

from llmproxy.utils import categorization
from llmproxy.utils.enums import BaseEnum
from llmproxy.utils.exceptions.llmproxy_client import (
    LLMProxyConfigError,
    ModelRequestFailed,
    RequestsFailed,
    UserConfigError,
)
from llmproxy.utils.exceptions.provider import UnsupportedModel
from llmproxy.utils.log import logger
from llmproxy.utils.sorting import MinHeap


def _get_settings_from_yml(
    path_to_yml: str = "",
) -> Dict[str, Any]:
    """Returns all of the data in the yaml file"""
    try:
        with open(path_to_yml, "r", encoding="utf-8") as file:
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

            # Loop through and aggregate all of the variations of "models" of each provider
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


def _setup_user_models(available_models=None, settings=None) -> Dict[str, object]:
    """Setup all available models and return dict of {name: instance_of_model}"""
    if not available_models:
        raise UserConfigError(
            "Available models not found, please ensure you have the latest version of LLM Proxy."
        )
    if not settings:
        raise UserConfigError(
            "Configuration not found, please ensure that you the correct path and format of configuration file"
        )
    if not settings["user_settings"]:
        raise UserConfigError(
            "No models found in user settings. Please ensure the format of the configuration file is correct."
        )

    try:
        user_models = {}
        # Compare user models with available_models
        for provider in settings["user_settings"]:
            model_name = provider["model"].lower().strip()
            # Check if user model in available models

            if model_name in available_models:
                # If the user providers NO variations then raise error
                if "models" not in provider or provider["models"] is None:
                    raise LLMProxyConfigError(
                        f"No models provided in llmproxy.config.yml for the following model: {model_name}"
                    )

                # Loop through and set up instance of model
                for model in provider["models"]:
                    # Different setup for vertexai
                    if model not in available_models[model_name]["models"]:
                        raise UnsupportedModel(
                            f"{model} is not available, yet!",
                            error_type="UnsupportedModel",
                        )

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
    except UnsupportedModel as e:
        raise e
    except Exception as e:
        raise UserConfigError(
            f"Unknown error occured during llmproxy.config setup:{e}"
        ) from e


@dataclass
class CompletionResponse:
    """
    response: Data on successful response else ""
    errors: List of all models and exceptions - if raised
    """

    response: str = ""
    errors: List = field(default_factory=list)


class RouteType(str, BaseEnum):
    COST = "cost"
    CATEGORY = "category"


class LLMProxy:
    def __init__(
        self,
        path_to_user_configuration: str = "llmproxy.config.yml",
        path_to_env_vars: str = ".env",
    ) -> None:
        self.user_models = {}
        self.route_type = "cost"

        load_dotenv(path_to_env_vars)

        # Read YML and see which models the user wants
        dev_settings = _get_settings_from_yml(
            path_to_yml="llmproxy/config/internal.config.yml"
        )
        user_settings = _get_settings_from_yml(path_to_yml=path_to_user_configuration)

        # Setup available models
        available_models = _setup_available_models(settings=dev_settings)
        self.user_models = _setup_user_models(
            settings=user_settings, available_models=available_models
        )
        self.available_models = available_models

    def route(
        self,
        route_type: Literal["cost", "category"] = RouteType.COST.value,
        prompt: str = "",
    ) -> CompletionResponse:
        match RouteType(route_type.lower()):
            case RouteType.COST:
                return self._cost_route(prompt=prompt)
            case RouteType.CATEGORY:
                return self._category_route(prompt=prompt)
            case _:
                raise ValueError("Invalid route type, please try again")

    def _cost_route(self, prompt: str):
        min_heap = MinHeap()
        for model_name, instance in self.user_models.items():
            try:
                logger.info(msg="========Start Cost Estimation===========")
                cost = instance.get_estimated_max_cost(prompt=prompt)
                logger.info(msg="========End Cost Estimation===========\n")

                item = {"name": model_name, "cost": cost, "instance": instance}
                min_heap.push(cost, item)
            except Exception as e:
                logger.error(msg=e)

        completion_res = None
        errors = []
        while not completion_res:
            # Iterate through heap until there are no more options
            min_val_instance = min_heap.pop_min()
            if not min_val_instance:
                break

            instance_data = min_val_instance["data"]
            logger.info("Making request to model: %s\n", instance_data["name"])
            logger.info("ROUTING...\n")

            # Attempt to make request to model
            try:
                completion_res = instance_data["instance"].get_completion(prompt=prompt)
                logger.info(
                    "==========ROUTING COMPLETE! Call to model successful!==========\n"
                )
            except Exception as e:
                errors.append({"model_name": instance_data["name"], "error": e})
                logger.warning("Request to model %s failed!\n", instance_data["name"])
                logger.warning("Error when making request to model: %s\n", e)

        # If all model fails raise an Exception to notify user
        if not completion_res:
            raise ModelRequestFailed(
                "Requests to all models failed! Please check your configuration!"
            )

        return CompletionResponse(response=completion_res, errors=errors)

    def _category_route(self, prompt: str):
        min_heap = MinHeap()
        best_fit_category = categorization.categorize_text(prompt)
        for (
            model_name,
            instance,
        ) in self.user_models.items():
            logger.info(
                msg="========Start fetching model for category routing==========="
            )
            category_rank = instance.get_category_rank(best_fit_category)
            item = {"name": model_name, "rank": category_rank, "instance": instance}
            min_heap.push(category_rank, item)
            logger.info(msg="Sorting fetched models based on proficency...")
            logger.info(
                msg="========Finished fetching model for category routing=============\n"
            )

        completion_res = None
        errors = []
        while not completion_res:
            # Iterate through heap until there are no more options
            min_val_instance = min_heap.pop_min()
            if not min_val_instance:
                break

            instance_data = min_val_instance["data"]
            logger.info("Making request to model: %s", instance_data["name"])
            logger.info("ROUTING...")

            try:
                completion_res = instance_data["instance"].get_completion(prompt=prompt)
                logger.info("ROUTING COMPLETE! Call to model successful!\n")
            except Exception as e:
                errors.append({"model_name": instance_data["name"], "error": e})
                logger.warning("Request to model %s failed!\n", instance_data["name"])
                logger.warning("Error when making request to model: %s\n", e)

        if not completion_res:
            raise RequestsFailed(
                "Requests to all models failed! Please check your configuration!"
            )

        return CompletionResponse(response=completion_res, errors=errors)
