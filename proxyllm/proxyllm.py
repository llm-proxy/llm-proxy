import importlib
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal

import yaml
from dotenv import load_dotenv

from proxyllm.config.internal_config import internal_config
from proxyllm.provider.base import BaseAdapter
from proxyllm.utils import categorization, proxy_logger
from proxyllm.utils.cost import calculate_estimated_max_cost
from proxyllm.utils.enums import BaseEnum
from proxyllm.utils.exceptions.llmproxy_client import (
    LLMProxyConfigError,
    ModelRequestFailed,
    RequestsFailed,
    UserChatHistoryError,
    UserConfigError,
)
from proxyllm.utils.exceptions.provider import UnsupportedModel
from proxyllm.utils.sorting import MinHeap


@dataclass
class CompletionResponse:
    """
    Data structure to store the response from a model completion request.

    Attributes:
        response (str): The response text from the model, if successful; otherwise, an empty string.
        response_model (str): The model that successfully responded to the request.
        errors (List): A list of error messages from models that failed to respond.
        chat_history (List): A list of chat messages/responses from the user and models.
    """

    response: str = ""
    response_model: str = ""
    errors: List = field(default_factory=list)
    chat_history: List[Dict[str, str]] = field(default_factory=list)


class RouteType(str, BaseEnum):
    """
    Enumeration for route types supported by LLM Proxy.

    Attributes:
        COST: Route requests based on cost efficiency.
        CATEGORY: Route requests based on category elo.
    """

    COST = "cost"
    CATEGORY = "category"
    ELO = "elo"


def _get_settings_from_yml(
    path_to_yml: str = "",
) -> Dict[str, Any]:
    """
    Load and return settings from a YAML file.

    Args:
        path_to_yml (str): File path to the YAML configuration file.

    Returns:
        Dict[str, Any]: Configuration settings from the YAML file.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        yaml.YAMLError: If there is an error parsing the YAML file.
    """
    try:
        with open(path_to_yml, "r", encoding="utf-8") as file:
            result = yaml.safe_load(file)
            return result
    except (FileNotFoundError, yaml.YAMLError) as e:
        raise e


def _setup_available_models(settings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create a mapping of available models from settings.

    Args:
        settings (List[Dict[str, Any]]): Configuration settings that include model information.

    Returns:
        Dict[str, Any]: A dictionary mapping model names to their corresponding adapter instances and model data.

    Raises:
        ImportError: If there is an error importing the adapter module or class.
    """
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
    """
    Setup and return user-specified models based on configuration.

    Args:
        available_models (Dict[Any, Any]): Dictionary of available model adapters.
        yml_settings (Dict[Any, Any]): User-specified settings from YAML configuration.
        constructor_settings (Dict[Any, Any]): Additional settings provided at runtime.

    Returns:
        Dict[str, BaseAdapter]: A dictionary mapping model names to their instantiated adapter objects.

    Raises:
        UserConfigError: If there is a misconfiguration in user settings.
    """

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
        if not constructor_settings:
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
                    model_lower = (
                        model.lower()  # Do this to avoid case errors with user inputs
                    )
                    if model_lower not in available_models[provider_name]["models"]:
                        raise UnsupportedModel(
                            f"{model} is not available, yet!",
                            error_type="UnsupportedModel",
                        )

                    # Common params among all models
                    common_parameters = {
                        "max_output_tokens": provider["max_output_tokens"],
                        "temperature": provider["temperature"],
                        "model": model_lower,
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

                    user_models[model_lower] = model_instance

        return user_models
    except UnsupportedModel as e:
        raise e
    except Exception as e:
        raise UserConfigError(
            f"Unknown error occured during llmproxy.config setup:{e}"
        ) from e


def _setup_model_data(settings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract and return the cost data for each model from configuration settings.

    Args:
        settings (List[Dict[str, Any]]): List of configuration settings for each model.

    Returns:
        Dict[str, Any]: A dictionary mapping model names to their cost data.

    Raises:
        KeyError: If expected keys are missing from the settings.
    """
    try:
        model_data = {}
        # Loop through all the providers in settings
        for provider in settings:
            # Loop through the "models" and save the cost data for each model
            for model_yml_data in provider.get("models", []):
                (
                    model_name,
                    prompt_cost,
                    completion_cost,
                    model_elo,
                ) = model_yml_data.values()
                model_data[model_name] = {
                    "prompt": prompt_cost,
                    "completion": completion_cost,
                    "elo": model_elo,
                }
        return model_data
    except Exception as e:
        raise e


def _get_route_type(
    user_settings: Dict[str, Any] | None,
    constructor_route_type: Literal["cost", "category", "elo"] | None,
) -> Literal["cost", "category", "elo"]:
    """
    Determine the routing type based on user settings or constructor arguments.

    Args:
        user_settings (Dict[str, Any] | None): Configuration from the user settings.
        constructor_route_type (Literal["cost", "category", "elo"] | None): Routing type specified at object construction.

    Returns:
        Literal["cost", "category", "elo"]: The determined route type.

    Raises:
        UserConfigError: If the route type is not specified or invalid.
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


class LLMProxy:
    """
    Main class for LLM Proxy to route requests.

    This class initializes the proxy with user-defined settings, sets up model adapters,
    and routes requests to the most suitable models based on the selected routing strategy.

    Attributes:
        user_models (Dict[str, BaseAdapter]): Models configured by the user, ready for use.
        available_models (Dict[str, Any]): All models available within the proxy.
        route_type (Literal["cost", "category"]): Selected routing strategy.
        model_data (Dict[str, Any]): Data for each model used in cost-based or elo routing.
    """

    def __init__(
        self,
        path_to_user_configuration: str = "llmproxy.config.yml",
        path_to_env_vars: str = ".env",
        route_type: Literal["cost", "category", "elo"] | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize YourClass instance.

        Args:
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

        if self.route_type == RouteType.COST or RouteType.ELO:
            # Setup the cost data of each model
            self.model_data = _setup_model_data(settings=internal_config)

    def route(
        self,
        prompt: str = "",
        chat_history: List[Dict[str, str]] = None,
    ) -> CompletionResponse:
        """
        Routes the request to the appropriate models based on the routing strategy.

        Args:
            prompt (str): The input prompt to generate text for.
            chat_history (List[Dict[str, str]]): The chat history for conversation

        Returns:
            CompletionResponse: The generated text response along with any errors encountered.
        """
        if isinstance(chat_history, list) or chat_history is None:
            match RouteType(self.route_type):
                case RouteType.COST:
                    return self._cost_route(prompt=prompt, chat_history=chat_history)
                case RouteType.CATEGORY:
                    return self._category_route(
                        prompt=prompt, chat_history=chat_history
                    )
                case RouteType.ELO:
                    return self._elo_route(prompt=prompt, chat_history=chat_history)
                case _:
                    raise ValueError("Invalid route type, please try again")
        else:
            raise UserChatHistoryError(
                "Incorrect format for chat_history: chat_history needs to be a List[Dict[str,str]]."
            )

    def _cost_route(self, prompt: str, chat_history: List[Dict[str, str]]):
        """
        Routes requests based on cost efficiency.

        Args:
            prompt (str): The input prompt for the text generation.

        Returns:
            CompletionResponse: The response from the most cost-effective model.
        """

        if not prompt or prompt.isspace():
            raise ValueError("No prompt provided.")

        min_heap = MinHeap()
        provider_token_data = {}

        for model_name, instance in self.user_models.items():
            try:
                proxy_logger.log(msg="========Start Cost Estimation===========")

                proxy_logger.log(msg=f"MODEL: {model_name}", color="PURPLE")
                proxy_logger.log(
                    msg=f"PROMPT (COST/CHARACTER): {self.model_data[model_name]['prompt']}"
                )
                proxy_logger.log(
                    msg=f"PROMPT (COST/CHARACTER): {self.model_data[model_name]['completion']}"
                )

                instance_provider = instance.__class__.__name__

                # Save token data per provider
                if instance_provider not in provider_token_data:
                    provider_token_data[instance_provider] = instance.tokenize(
                        prompt=prompt
                    )

                token_data = provider_token_data[instance_provider]

                cost = calculate_estimated_max_cost(
                    price_data=self.model_data[model_name],
                    num_of_input_tokens=token_data.num_of_input_tokens,
                    max_output_tokens=token_data.num_of_output_tokens,
                )

                item = {
                    "name": model_name,
                    "cost": cost,
                    "instance": instance,
                }

                min_heap.push(cost, item)

                proxy_logger.log(msg=f"INPUT TOKENS: {token_data.num_of_input_tokens}")
                proxy_logger.log(
                    msg=f"COMPLETION TOKENS: {token_data.num_of_output_tokens}"
                )

                proxy_logger.log(msg=f"COST: {cost}", color="GREEN")
                proxy_logger.log(msg="========End Cost Estimation===========\n")
            except Exception as e:
                proxy_logger.log(level="ERROR", msg=str(e))
                proxy_logger.log(level="ERROR", msg="(¬_¬)", file_logger_on=False)
                proxy_logger.log(msg="========End Cost Estimation===========\n")

        completion_res = None
        errors = []
        response_model = ""

        while not completion_res:
            # Iterate through heap until there are no more options
            min_val_instance = min_heap.pop_min()
            if not min_val_instance:
                break

            instance_data = min_val_instance["data"]
            proxy_logger.log(msg="========START COST ROUTING===========")
            proxy_logger.log(msg=f"Making request to model:{instance_data['name']}")
            proxy_logger.log(msg="ROUTING...")

            try:
                completion_res = instance_data["instance"].get_completion(
                    prompt=prompt, chat_history=chat_history
                )
                response_model = instance_data["name"]
                proxy_logger.log(
                    msg="==========COST ROUTING COMPLETE! Call to model successful!==========",
                    color="GREEN",
                )
                proxy_logger.log(msg="(• ◡ •)\n", file_logger_on=False, color="GREEN")
            except Exception as e:
                errors.append(
                    {
                        "model_name": instance_data["name"],
                        "error_type": e.__class__.__name__,
                        "error": str(e),
                    }
                )

                proxy_logger.log(
                    level="ERROR",
                    msg=f"Request to model {instance_data['name']} failed!",
                )
                proxy_logger.log(
                    level="ERROR", msg=f"Error when making request to model: {e}"
                )
                proxy_logger.log(level="ERROR", msg="(•᷄ ∩ •᷅)", file_logger_on=False)

                proxy_logger.log(
                    level="ERROR",
                    msg="========COST ROUTING FAILED!===========\n",
                )

        # If all model fails raise an Exception to notify user
        if not completion_res:
            raise ModelRequestFailed(
                "Requests to all models failed! Please check your configuration!"
            )

        return CompletionResponse(
            response=completion_res["response"],
            response_model=response_model,
            errors=errors,
            chat_history=completion_res["chat_history"],
        )

    def _category_route(self, prompt: str, chat_history: List[Dict[str, str]]):
        """
        Routes requests based on the category elo of available models.

        Args:
            prompt (str): The input prompt for the text generation.

        Returns:
            CompletionResponse: The response from the model best suited for the prompt's category.
        """
        min_heap = MinHeap()
        best_fit_category = categorization.categorize_text(prompt)
        proxy_logger.log(msg=f"CATEGORY OF PROMPT: {best_fit_category}")

        for model_name, instance in self.user_models.items():
            proxy_logger.log(
                msg="========Fetching Model For Category Routing===========",
            )

            category_rank = instance.get_category_rank(best_fit_category)
            item = {"name": model_name, "rank": category_rank, "instance": instance}
            min_heap.push(category_rank, item)

            proxy_logger.log(
                msg="========Finished Fetching Model For Category Routing=============\n",
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
            proxy_logger.log(msg="========START COST ROUTING===========")
            proxy_logger.log(
                msg=f"Making request to model: {instance_data['name']}",
            )

            try:
                completion_res = instance_data["instance"].get_completion(
                    prompt=prompt, chat_history=chat_history
                )
                response_model = instance_data["name"]
                proxy_logger.log(
                    msg="CATEGORY ROUTING COMPLETE! Call to model successful!",
                )
                proxy_logger.log(msg="(• ◡ •)\n", file_logger_on=False, color="GREEN")
            except Exception as e:
                errors.append(
                    {
                        "model_name": instance_data["name"],
                        "error_type": e.__class__.__name__,
                        "error": str(e),
                    }
                )

                proxy_logger.log(
                    level="ERROR",
                    msg=f"Request to model {instance_data['name']} failed!",
                )

                proxy_logger.log(
                    level="ERROR",
                    msg=f"Error when making request to model: {e}",
                )

                proxy_logger.log(level="ERROR", msg="(•᷄ ∩ •᷅)", file_logger_on=False)

                proxy_logger.log(
                    level="ERROR",
                    msg="========CATEGORY ROUTING FAILED!===========\n",
                )

        if not completion_res:
            raise RequestsFailed(
                "Requests to all models failed! Please check your configuration!"
            )

        return CompletionResponse(
            response=completion_res["response"],
            response_model=response_model,
            errors=errors,
            chat_history=completion_res["chat_history"],
        )

    def _elo_route(self, prompt: str, chat_history: List[Dict[str, str]]):
        """
        Routes the request to the appropriate models based on elo elo rating of available models

        Args:
            prompt (str): The input prompt to generate text for.

        Returns:
            CompletionResponse: The generated text response along with any errors encountered.
        """

        min_heap = MinHeap()
        proxy_logger.log(
            msg="Sorting fetched models based on elo rating...", color="GREEN"
        )
        for model_name, instance in self.user_models.items():
            proxy_logger.log(
                msg="========Fetching Models for Elo Routing===========",
            )

            proxy_logger.log(msg=f"MODEL: {model_name}", color="PURPLE")

            elo_rating = self.model_data[model_name]["elo"]

            proxy_logger.log(msg=f"ELO RATING OF MODEL: {elo_rating}", color="BLUE")

            item = {"name": model_name, "elo": elo_rating, "instance": instance}
            min_heap.push(-1 * elo_rating, item)

            proxy_logger.log(
                msg="========Finished Model Fetching For Elo Routing=============\n",
            )

        completion_res = None
        errors = []
        response_model = ""
        while not completion_res:
            # Iterate through heap until there are no more options
            max_val_instance = min_heap.pop_min()
            if not max_val_instance:
                break

            instance_data = max_val_instance["data"]
            proxy_logger.log(msg="========START ELO ROUTING===========")
            proxy_logger.log(msg=f"Making request to model:{instance_data['name']}")
            proxy_logger.log(msg="ROUTING...")

            try:
                completion_res = instance_data["instance"].get_completion(
                    prompt=prompt, chat_history=chat_history
                )
                response_model = instance_data["name"]
                proxy_logger.log(
                    msg="==========ELO ROUTING COMPLETE! Call to model successful!==========",
                    color="GREEN",
                )
                proxy_logger.log(msg="(• ◡ •)\n", file_logger_on=False, color="GREEN")
            except Exception as e:
                errors.append(
                    {
                        "model_name": instance_data["name"],
                        "error_type": e.__class__.__name__,
                        "error": str(e),
                    }
                )

                proxy_logger.log(
                    level="ERROR",
                    msg=f"Request to model {instance_data['name']} failed!",
                )

                proxy_logger.log(
                    level="ERROR",
                    msg=f"Error when making request to model: {e}",
                )

                proxy_logger.log(level="ERROR", msg="(•᷄ ∩ •᷅)", file_logger_on=False)

                proxy_logger.log(
                    level="ERROR",
                    msg="========ELO ROUTING FAILED!===========\n",
                )

        if not completion_res:
            raise RequestsFailed(
                "Requests to all models failed! Please check your configuration!"
            )

        return CompletionResponse(
            response=completion_res["response"],
            response_model=response_model,
            errors=errors,
            chat_history=completion_res["chat_history"],
        )
