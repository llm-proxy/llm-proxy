import importlib
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal

import yaml
from dotenv import load_dotenv

from llmproxy.config.internal_config import internal_config
from llmproxy.provider.base import BaseAdapter
from llmproxy.utils import categorization
from llmproxy.utils.enums import BaseEnum
from llmproxy.utils.exceptions.llmproxy_client import (
    LLMProxyConfigError,
    ModelRequestFailed,
    RequestsFailed,
    UserConfigError,
)
from llmproxy.utils.exceptions.provider import UnsupportedModel
from llmproxy.utils.log import CustomLogger, console_logger, file_logger
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


def _setup_available_models(settings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Returns classname with list of available_models for provider"""
    try:
        available_models = {}
        # Loop through each provider
        for provider in settings:
            key = provider["provider"].lower()
            import_path = provider["adapter_path"]

            # Loop through and aggregate all of the variations of "models" of each provider
            provider_models = set()
            for model in provider.get("models", []):
                provider_models.add(model["name"])

            module_name, class_name = import_path.rsplit(".", 1)

            module = importlib.import_module(module_name)
            model_class = getattr(module, class_name)

            # return dict with class path and models set, with all of the variations/models of that provider
            available_models[key] = {
                "adapter_instance": model_class,
                "models": provider_models,
            }

        return available_models
    except Exception as e:
        raise e


def _setup_user_models(
    available_models=None, yml_settings=None, constructor_settings=None
) -> Dict[str, BaseAdapter]:
    """Setup all available models and return dict of {name: instance_of_model}"""

    if not available_models:
        raise UserConfigError(
            "Available models not found, please ensure you have the latest version of LLM Proxy."
        )
    if not yml_settings:
        raise UserConfigError(
            "Configuration not found, please ensure that you the correct path and format of configuration file"
        )
    if not yml_settings["provider_settings"]:
        raise UserConfigError(
            "No models found in user settings. Please ensure the format of the configuration file is correct."
        )

    try:
        # optional_config = yml_settings.get("optional_configuration", constructor_settings) or {}
        optional_config = (
            yml_settings.get("optional_configuration", None)
            or constructor_settings
            or {}
        )
        user_models = {}
        # Compare user models with available_models
        for provider in yml_settings["provider_settings"]:
            model_name = provider["provider"].lower().strip()
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
                        "timeout": optional_config.get("timeout", None),
                    }

                    # Different setup for vertexai
                    if model_name == "vertexai":
                        common_parameters.update(
                            {
                                # Project ID required for VertexAI
                                "project_id": os.getenv(provider["project_id_var"]),
                                # No internal timeout flag provided
                                "force_timeout": optional_config.get(
                                    "force_timeout", False
                                ),
                            }
                        )
                    else:
                        common_parameters["api_key"] = os.getenv(
                            provider["api_key_var"]
                        )

                    model_instance = available_models[model_name]["adapter_instance"](
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


def _get_route_type(
    user_settings: Dict[str, Any] | None,
    constructor_route_type: Literal["cost", "category"] | None,
) -> Literal["cost", "category"]:
    """
    Get the route type from user settings or constructor parameters.

    Args:
        user_settings (Optional[Dict[str, Any]]): User settings containing proxy configuration.
            If None, the route type will default to constructor_route_type.
        constructor_route_type (Optional[Literal["cost", "category"]]): Route type specified during object construction.

    Raises:
        ValueError: If no route type is specified in either user settings or constructor parameters.

    """
    if user_settings and user_settings.get("proxy_configuration", None):
        route_type = (
            user_settings.get("proxy_configuration", None).get("route_type", None)
            or constructor_route_type
        )
    else:
        route_type = constructor_route_type

    if route_type:
        return route_type

    raise ValueError(
        "No route type was specified, please add the route_type in the llmproxy yaml config or LLMProxy constructor"
    )


@dataclass
class CompletionResponse:
    """
    response: Data on successful response else ""
    errors: List of all models and exceptions - if raised
    """

    response: str = ""
    response_model: str = ""
    errors: List = field(default_factory=list)


class RouteType(str, BaseEnum):
    COST = "cost"
    CATEGORY = "category"


class LLMProxy:
    def __init__(
        self,
        path_to_user_configuration: str = "llmproxy.config.yml",
        path_to_env_vars: str = ".env",
        route_type: Literal["cost", "category"] | None = None,
        timeout: int | None = None,
        force_timeout: bool = False,
    ) -> None:
        load_dotenv(path_to_env_vars)
        # Read YML for user settings
        user_settings = _get_settings_from_yml(path_to_yml=path_to_user_configuration)

        # Setup user cost
        self.route_type = _get_route_type(
            user_settings=user_settings, constructor_route_type=route_type
        )

        # Setup available models
        available_models = _setup_available_models(settings=internal_config)

        # Setup user models
        self.user_models: Dict[str, BaseAdapter] = _setup_user_models(
            yml_settings=user_settings,
            available_models=available_models,
            constructor_settings={"timeout": timeout, "force_timeout": force_timeout},
        )

        self.available_models = available_models

    def route(
        self,
        prompt: str = "",
    ) -> CompletionResponse:
        match RouteType(self.route_type):
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
                file_logger.info(msg="========Start Cost Estimation===========")
                console_logger.info(msg="========Start Cost Estimation===========")

                cost = instance.get_estimated_max_cost(prompt=prompt)
                file_logger.info(msg="========End Cost Estimation===========\n")
                console_logger.info(msg="========End Cost Estimation===========\n")

                item = {"name": model_name, "cost": cost, "instance": instance}
                min_heap.push(cost, item)
            except Exception as e:
                file_logger.error(msg=e)
                console_logger.error(msg=e)
                console_logger.error("(¬_¬)")
                file_logger.info(msg="========End Cost Estimation===========\n")
                console_logger.info(msg="========End Cost Estimation===========\n")
        completion_res = None
        errors = []
        response_model = ""
        while not completion_res:
            # Iterate through heap until there are no more options
            min_val_instance = min_heap.pop_min()
            if not min_val_instance:
                break

            instance_data = min_val_instance["data"]
            file_logger.info(msg="========START COST ROUTING===========")
            console_logger.info(msg="========START COST ROUTING===========")
            file_logger.info(f"Making request to model:{instance_data['name']}")
            console_logger.info(f"Making request to model:{instance_data['name']}")
            file_logger.info("ROUTING...")
            console_logger.info("ROUTING...")
            # CustomLogger.loading_animation_sucess()

            # Attempt to make request to model
            try:
                completion_res = instance_data["instance"].get_completion(prompt=prompt)
                # CustomLogger.loading_animation_sucess()

                response_model = instance_data["name"]
                console_logger.info(
                    CustomLogger.CustomFormatter.green
                    + "(• ◡ •)"
                    + CustomLogger.CustomFormatter.reset
                )
                file_logger.info(
                    "==========COST ROUTING COMPLETE! Call to model successful!==========\n"
                )
                console_logger.info(
                    "==========COST ROUTING COMPLETE! Call to model successful!==========\n"
                )
            except Exception as e:
                ## CustomLogger.loading_animation_failure()
                errors.append({"model_name": instance_data["name"], "error": e})

                file_logger.error(f"Request to model {instance_data['name']} failed!")
                console_logger.error(
                    f"Request to model {instance_data['name']} failed!"
                )
                file_logger.error(f"Error when making request to model: {e}")
                console_logger.error(f"Error when making request to model: {e}")
                console_logger.error("(•᷄ ∩ •᷅)")
                file_logger.info(msg="========COST ROUTING FAILED!===========\n")
                console_logger.info(msg="========COST ROUTING FAILED!===========\n")

        # If all model fails raise an Exception to notify user
        if not completion_res:
            raise ModelRequestFailed(
                "Requests to all models failed! Please check your configuration!"
            )

        return CompletionResponse(
            response=completion_res, response_model=response_model, errors=errors
        )

    def _category_route(self, prompt: str):
        min_heap = MinHeap()
        best_fit_category = categorization.categorize_text(prompt)
        for (
            model_name,
            instance,
        ) in self.user_models.items():
            file_logger.info(
                msg="========Start fetching model for category routing==========="
            )
            console_logger.info(
                msg="========Start fetching model for category routing==========="
            )

            category_rank = instance.get_category_rank(best_fit_category)
            item = {"name": model_name, "rank": category_rank, "instance": instance}
            min_heap.push(category_rank, item)
            file_logger.info(msg="Sorting fetched models based on proficency...")
            console_logger.info(msg="Sorting fetched models based on proficency...")
            file_logger.info(
                msg="========Finished fetching model for category routing=============\n"
            )
            console_logger.info(
                msg="========Finished fetching model for category routing=============\n"
            )

        completion_res = None
        errors = []
        response_model = ""
        while not completion_res:
            # Iterate through heap until there are no more options
            min_val_instance = min_heap.pop_min()
            if not min_val_instance:
                break

            instance_data = min_val_instance["data"]
            file_logger.info(f"Making request to model: {instance_data['name']}")
            console_logger.info(f"Making request to model: {instance_data['name']}")

            try:
                completion_res = instance_data["instance"].get_completion(prompt=prompt)
                response_model = instance_data["name"]
                file_logger.info(
                    "CATEGORY ROUTING COMPLETE! Call to model successful!\n"
                )
                console_logger.info(
                    "CATEGORY ROUTING COMPLETE! Call to model successful!"
                )
                console_logger.info(
                    CustomLogger.CustomFormatter.green
                    + "(• ◡ •)\n"
                    + CustomLogger.CustomFormatter.reset
                )
            except Exception as e:
                errors.append({"model_name": instance_data["name"], "error": e})
                file_logger.error("Request to model %s failed!", instance_data["name"])
                console_logger.error(
                    "Request to model %s failed!", instance_data["name"]
                )
                file_logger.error("Error when making request to model: %s\n", e)
                console_logger.error("Error when making request to model: %s", e)
                console_logger.error("(•᷄ ∩ •᷅)\n")

        if not completion_res:
            raise RequestsFailed(
                "Requests to all models failed! Please check your configuration!"
            )

        return CompletionResponse(
            response=completion_res, response_model=response_model, errors=errors
        )
