import importlib
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal

import yaml
from dotenv import load_dotenv

from llmproxy.config.internal_config import internal_config
from llmproxy.provider.base import BaseAdapter
from llmproxy.utils import categorization, logger
from llmproxy.utils.enums import BaseEnum
from llmproxy.utils.exceptions.llmproxy_client import (
    LLMProxyConfigError,
    ModelRequestFailed,
    RequestsFailed,
    UserConfigError,
)
from llmproxy.utils.exceptions.provider import UnsupportedModel
from llmproxy.utils.sorting import MinHeap


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
            model_costs = {}
            for model in provider.get("models", []):
                provider_models.add(model["name"])

            module_name, class_name = import_path.rsplit(".", 1)

            module = importlib.import_module(module_name)
            model_class = getattr(module, class_name)

            # return dict with class path and models set and model cost data, with all of the variations/models of that provider
            available_models[key] = {
                "adapter_instance": model_class,
                "models": provider_models,
            }

        return available_models
    except Exception as e:
        raise e


def _setup_user_models(
    available_models: Dict[Any, Any],
    yml_settings: Dict[Any, Any],
    constructor_settings: Dict[Any, Any] | None = None,
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
        # Return dict
        user_models = {}
        optional_config = constructor_settings
        if constructor_settings is None:
            optional_config = yml_settings.get("optional_configuration", None) or {}

        # Compare user models with available_models
        for provider in yml_settings["provider_settings"]:
            provider_name = provider["provider"].lower().strip()

            # Check if provider is in available models
            if provider_name in available_models:
                # If the user providers NO variations then raise error
                if "models" not in provider or provider["models"] is None:
                    raise LLMProxyConfigError(
                        f"No models provided in llmproxy.config.yml for the following model: {provider_name}"
                    )

                # Loop through user's provider's models and set up instance of model if available
                for model in provider["models"]:
                    if model not in available_models[provider_name]["models"]:
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
                    if provider_name == "vertexai":
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

                    model_instance = available_models[provider_name][
                        "adapter_instance"
                    ](**common_parameters)
                    user_models[model] = model_instance

        return user_models
    except UnsupportedModel as e:
        raise e
    except Exception as e:
        raise UserConfigError(
            f"Unknown error occured during llmproxy.config setup:{e}"
        ) from e


def _setup_models_cost_data(settings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extracts cost data for each model from the given list of settings dictionaries.

    Args:
    - settings: A list of dictionaries, each containing information about models and their costs.

    Returns:
    A dictionary containing model names as keys and their associated cost data as values.

    Raises:
    Exception: If there's any error during processing.
    """
    try:
        models_cost_data = {}
        # Loop through all the providers in settings
        for provider in settings:
            # Loop through the "models" and save the cost data for each model
            for model_data in provider.get("models", []):
                model_name, prompt_cost, completion_cost = model_data.values()
                models_cost_data[model_name] = {
                    "prompt": prompt_cost,
                    "completion": completion_cost,
                }
        return models_cost_data
    except Exception as e:
        raise e


def _get_route_type(
    user_settings: Dict[str, Any] | None,
    constructor_route_type: Literal["cost", "category"] | None,
) -> Literal["cost", "category"]:
    """
    Get the route type from constructor parameters or user settings.

    Args:
        user_settings (Optional[Dict[str, Any]]): User settings containing proxy configuration.
            If None, the route type will default to constructor_route_type.
        constructor_route_type (Optional[Literal["cost", "category"]]): Route type specified during object construction.

    Returns:
        Literal["cost", "category"]: The selected route type.

    Raises:
        ValueError: If no route type is specified in either user settings or constructor parameters.
    """
    route_type = None
    if constructor_route_type is not None:
        route_type = constructor_route_type
    elif user_settings is not None and isinstance(
        user_settings.get("proxy_configuration"), dict
    ):
        proxy_configuration = user_settings.get("proxy_configuration", {})
        route_type = proxy_configuration.get("route_type", None)

    else:
        raise UserConfigError(
            "No route type was specified. Please add the route_type in the llmproxy yaml config or LLMProxy constructor."
        )

    if route_type not in RouteType.list_values():
        raise UserConfigError(
            f"Invalid route type, please try ensure you have configured one of the follow routes: {', '.join(RouteType.list_values())}"
        )

    return route_type

class Config:
    def __init__(self):
        self.config_cache = {}
        self.mod_times = {}
    
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
    

class LLMProxy:
    def __init__(
        self,
        path_to_user_configuration: str = "llmproxy.config.yml",
        path_to_env_vars: str = ".env",
        route_type: Literal["cost", "category"] | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize YourClass instance.

        Parameters:
            path_to_user_configuration (str): Path to user configuration YAML file.
            path_to_env_vars (str): Path to environment variables file.
            route_type (Literal["cost", "category"] | None): Type of route.
            timeout (int): Timeout value in seconds (optional).
            force_timeout (bool): Whether to force timeout (optional).
            **kwargs: Additional, optional_configuration, keyword arguments for setting up user models.
                Only pass in optional_configuration paramters settings that you want to override
        """
        load_dotenv(path_to_env_vars)
        # Read YML for user settings
        user_settings = _get_settings_from_yml(path_to_yml=path_to_user_configuration)

        # Setup available models
        self.available_models = _setup_available_models(settings=internal_config)

        # Setup user models
        self.user_models: Dict[str, BaseAdapter] = _setup_user_models(
            yml_settings=user_settings,
            available_models=self.available_models,
            constructor_settings=kwargs,
        )

        # Setup user cost
        self.route_type = _get_route_type(
            user_settings=user_settings, constructor_route_type=route_type
        )

        if self.route_type == RouteType.COST:
            # Setup the cost data of each model
            self.models_cost_data = _setup_models_cost_data(settings=internal_config)

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
            # Load the cost data of the current model to get the estimate routing cost
            try:
                logger.log(msg="========Start Cost Estimation===========")
                model_price_data = self.models_cost_data[model_name]
                cost = instance.get_estimated_max_cost(
                    prompt=prompt, price_data=model_price_data
                )

                logger.log(msg="========End Cost Estimation===========\n")

                item = {"name": model_name, "cost": cost, "instance": instance}
                min_heap.push(cost, item)
            except Exception as e:
                logger.log(level="ERROR", msg=str(e))
                logger.log(level="ERROR", msg="(¬_¬)", file_logger_on=False)
                logger.log(msg="========End Cost Estimation===========\n")

        completion_res = None
        errors = []
        response_model = ""
        while not completion_res:
            # Iterate through heap until there are no more options
            min_val_instance = min_heap.pop_min()
            if not min_val_instance:
                break

            instance_data = min_val_instance["data"]
            logger.log(msg="========START COST ROUTING===========")
            logger.log(msg=f"Making request to model:{instance_data['name']}")
            logger.log(msg="ROUTING...")

            # Attempt to make request to model
            try:
                completion_res = instance_data["instance"].get_completion(prompt=prompt)
                # CustomLogger.loading_animation_sucess()

                response_model = instance_data["name"]
                logger.log(
                    msg="==========COST ROUTING COMPLETE! Call to model successful!==========",
                    color="GREEN",
                )
                logger.log(msg="(• ◡ •)\n", file_logger_on=False, color="GREEN")
            except Exception as e:
                ## CustomLogger.loading_animation_failure()
                errors.append({"model_name": instance_data["name"], "error": e})

                logger.log(
                    level="ERROR",
                    msg=f"Request to model {instance_data['name']} failed!",
                )
                logger.log(
                    level="ERROR", msg=f"Error when making request to model: {e}"
                )
                logger.log(level="ERROR", msg="(•᷄ ∩ •᷅)", file_logger_on=False)

                logger.log(
                    level="ERROR",
                    msg="========COST ROUTING FAILED!===========\n",
                )

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
        for model_name, instance in self.user_models.items():
            logger.log(
                msg="========Fetching model for category routing===========",
            )

            logger.log(
                msg="Sorting fetched models based on proficency...",
            )
            category_rank = instance.get_category_rank(best_fit_category)
            item = {"name": model_name, "rank": category_rank, "instance": instance}
            min_heap.push(category_rank, item)

            logger.log(
                msg="========Finished fetching model for category routing=============\n",
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
            logger.log(
                msg=f"Making request to model: {instance_data['name']}",
            )

            try:
                completion_res = instance_data["instance"].get_completion(prompt=prompt)
                response_model = instance_data["name"]
                logger.log(
                    msg="CATEGORY ROUTING COMPLETE! Call to model successful!",
                )
                logger.log(msg="(• ◡ •)\n", file_logger_on=False, color="GREEN")
            except Exception as e:
                errors.append({"model_name": instance_data["name"], "error": e})

                logger.log(
                    level="ERROR",
                    msg=f"Request to model {instance_data['name']} failed!",
                )

                logger.log(
                    level="ERROR",
                    msg=f"Error when making request to model: {e}",
                )

                logger.log(level="ERROR", msg="(•᷄ ∩ •᷅)", file_logger_on=False)

                logger.log(
                    level="ERROR",
                    msg="========CATEGORY ROUTING FAILED!===========\n",
                )

        if not completion_res:
            raise RequestsFailed(
                "Requests to all models failed! Please check your configuration!"
            )

        return CompletionResponse(
            response=completion_res, response_model=response_model, errors=errors
        )
