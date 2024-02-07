import os

import pytest
import yaml

from llmproxy.config.internal_config import internal_config
from llmproxy.llmproxy import (
    LLMProxy,
    UserConfigError,
    _get_settings_from_yml,
    _setup_available_models,
    _setup_user_models,
)

CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
path_to_env_test = ".env.test"


def test_empty_model() -> None:
    # This need to be the specific exception that it would raise
    # You need to merge main for the updated exceptions
    with pytest.raises(Exception, match="No models provided in llmproxy.config.yml"):
        LLMProxy(
            path_to_user_configuration=f"{CURRENT_DIRECTORY}/empty_model_test.yml",
            path_to_env_vars=".env.test",
        )


def test_no_user_setting(tmp_path) -> None:
    yml_content = """
    """
    yml_path = tmp_path / "test_settings.yml"
    with open(yml_path, "w", encoding="utf-8") as file:
        file.write(yml_content)
    text = "Configuration not found, please ensure that you the correct path and format of configuration file"
    with pytest.raises(UserConfigError, match=text):
        LLMProxy(
            path_to_user_configuration=yml_path,
            path_to_env_vars=path_to_env_test,
        )


def test_no_model_in_user_setting(tmp_path) -> None:
    yml_content = """
    user_settings:
    """
    yml_path = tmp_path / "test_settings.yml"
    with open(yml_path, "w", encoding="utf-8") as file:
        file.write(yml_content)
    text = "No models found in user settings. Please ensure the format of the configuration file is correct."
    with pytest.raises(UserConfigError, match=text):
        LLMProxy(path_to_user_configuration=yml_path, path_to_env_vars=path_to_env_test)


def test_invalid_model() -> None:
    with pytest.raises(Exception, match="test is not available"):
        LLMProxy(
            path_to_user_configuration=f"{CURRENT_DIRECTORY}/invalid_model_test.yml",
            path_to_env_vars=".env.test",
        )


def test_get_settings_from_yml(tmp_path) -> None:
    yml_content = """
    user_settings:
      - provider: OpenAI
        api_key_var: OPENAI_API_KEY
        max_output_tokens: 256
        temperature: 0.1
        models:
            - gpt-3.5-turbo-instruct
            - gpt-3.5-turbo-1106
            - gpt-4
            - gpt-4-32k
    """
    yml_path = tmp_path / "test_settings.yml"
    with open(yml_path, "w", encoding="utf-8") as file:
        file.write(yml_content)

    LLMProxy(path_to_user_configuration=yml_path, path_to_env_vars=path_to_env_test)


def test_get_settings_from_invalid_yml() -> None:
    invalid_yml_path = "invalid_settings.yml"
    with pytest.raises((FileNotFoundError, yaml.YAMLError)):
        LLMProxy(
            path_to_user_configuration=invalid_yml_path,
            path_to_env_vars=path_to_env_test,
        )


def test_setup_available_models() -> None:
    _setup_available_models(settings=internal_config)


# TODO: ADD TEST
def test_setup_user_models() -> None:
    path_to_user_configuration_test = f"{CURRENT_DIRECTORY}/test.yml"
    LLMProxy(
        path_to_user_configuration=path_to_user_configuration_test,
        path_to_env_vars=path_to_env_test,
    )


def test_no_available_model_UserConfigError() -> None:
    text = "Available models not found, please ensure you have the latest version of LLM Proxy."
    with pytest.raises(
        UserConfigError,
        match=text,
    ):
        _setup_user_models(available_models=None, settings=None)


def test_setup_user_models_no_setting_UserConfigError():
    with pytest.raises(
        UserConfigError,
        match="Configuration not found, please ensure that you the correct path and format of configuration file",
    ):
        test_available_model = _setup_available_models(settings=internal_config)
        _setup_user_models(available_models=test_available_model, settings=None)


def test_setup_user_models_empty_user_settings():
    with pytest.raises(
        UserConfigError,
        match="No models found in user settings. Please ensure the format of the configuration file is correct.",
    ):
        test_available_model = _setup_available_models(settings=internal_config)
        _setup_user_models(
            available_models=test_available_model, settings={"user_settings": []}
        )


def test_setup_user_models_no_variation() -> None:
    with pytest.raises(
        UserConfigError,
    ):
        test_available_model = _setup_available_models(settings=internal_config)

        _setup_user_models(
            available_models=test_available_model,
            settings={
                "user_settings": [
                    {
                        "provider": "OpenAI",
                        "api_key_var": "OPENAI_API_KEY",
                        "max_output_tokens": 256,
                        "temperature": 0.1,
                        "models": None,
                    }
                ]
            },
        )


def test_invalid_route_type() -> None:
    prompt = "what's 9+10?"
    with pytest.raises(ValueError, match="'interest' is not a valid RouteType"):
        test = LLMProxy(
            path_to_user_configuration=f"{CURRENT_DIRECTORY}/test.yml",
            path_to_env_vars=".env.test",
        )
        test.route(route_type="interest", prompt=prompt)
