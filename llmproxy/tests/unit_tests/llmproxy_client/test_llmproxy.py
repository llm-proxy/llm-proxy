from llmproxy.llmproxy import (
    _get_settings_from_yml,
    _setup_available_models,
    _setup_user_models,
)
import pytest
import yaml

import os

from llmproxy.llmproxy import (
    LLMProxy,
    UnsupportedModel,
    UserConfigError,
    LLMProxyConfigError,
)


CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
path_to_env_test = ".env.test"


# TODO: FIX
def test_empty_model() -> None:
    # This need to be the specific exception that it would raise
    # You need to merge main for the updated exceptions
    with pytest.raises(Exception, match="No models provided in llmproxy.config.yml"):
        # You need to use this
        LLMProxy(
            path_to_user_configuration=f"{CURRENT_DIRECTORY}/empty_model_test.yml",
            path_to_env_vars=".env.test",
        )

        # NOTE: These are technically class functions; You should not be using them directly like this
        # You can do this for specfic tests for these functions, but not for a general LLM Proxy test;
        # For that you need to initalize the LLMProxy, because that is where the logic lies

        # settings = _get_settings_from_yml(
        #     path_to_yml=f"{CURRENT_DIRECTORY}/empty_model_test.yml"
        # )
        # available_model = _setup_available_models(settings=settings)
        # user_model = _setup_user_models(
        #     settings=settings, available_models=available_model
        # )


def test_invalid_model() -> None:
    with pytest.raises(Exception, match="test is not available"):
        LLMProxy(
            path_to_user_configuration=f"{CURRENT_DIRECTORY}/invalid_model_test.yml",
            path_to_env_vars=".env.test",
        )


# TODO: ADD TEST
def test_get_settings_from_yml(tmp_path) -> None:
    yml_content = """
    available_models:
      - name: OpenAI
        class: llmproxy.provider.openai.chatgpt.OpenAI
        models:
          - name: model1
            cost_per_token_input: 0.000001
            cost_per_token_output: 0.000002
    user_settings:
      - model: OpenAI
        api_key_var: OPENAI_API_KEY
        max_output_tokens: 256
        temperature: 0.1
        models:
          - model1
    """
    yml_path = tmp_path / "test_settings.yml"
    with open(yml_path, "w", encoding="utf-8") as file:
        file.write(yml_content)

    # Test the function with the temporary YAML file
    result = _get_settings_from_yml(path_to_yml=str(yml_path))

    # Assertions
    assert "available_models" in result
    assert "user_settings" in result
    assert result["available_models"][0]["name"] == "OpenAI"
    assert result["user_settings"][0]["model"] == "OpenAI"
    assert result["user_settings"][0]["api_key_var"] == "OPENAI_API_KEY"
    assert result["user_settings"][0]["models"][0] == "model1"


def test_get_settings_from_invalid_yml(tmp_path) -> None:
    # create a temporary yaml file with invalid content
    invalid_yml_content = "invalid_yaml_content"
    invalid_yml_path = tmp_path / "invalid_settings.yml"
    with open(invalid_yml_path, "w", encoding="utf-8") as file:
        file.write(invalid_yml_content)

    # rest the function with the invalid yaml file
    with pytest.raises((FileNotFoundError, yaml.YAMLError)):
        _get_settings_from_yml(path_to_yml="invalid_setting.yml")


# TODO: ADD TEST
def test_setup_available_models() -> None:
    mock_settings = {
        "available_models": [
            {
                "name": "OpenAI",
                "class": "llmproxy.provider.openai.chatgpt.OpenAI",
                "models": [
                    {
                        "name": "gpt-3.5-turbo-1106",
                        "cost_per_token_input": 0.000001,
                        "cost_per_token_output": 0.000002,
                    },
                    {
                        "name": "gpt-3.5-turbo-instruct",
                        "cost_per_token_input": 0.0000015,
                        "cost_per_token_output": 0.000002,
                    },
                ],
            },
        ]
    }
    result = _setup_available_models(settings=mock_settings)

    # Assertions
    assert "openai" in result
    assert len(result["openai"]["models"]) == 2
    assert "llmproxy.provider.openai.chatgpt.OpenAI" in str(result["openai"]["class"])


# TODO: ADD TEST
def test_setup_user_models() -> None:
    available_models_setting = {
        "available_models": [
            {
                "name": "OpenAI",
                "class": "llmproxy.provider.openai.chatgpt.OpenAI",
                "models": [
                    {
                        "name": "gpt-3.5-turbo-1106",
                        "cost_per_token_input": 0.0000010,
                        "cost_per_token_output": 0.0000020,
                    },
                    {
                        "name": "gpt-3.5-turbo-instruct",
                        "cost_per_token_input": 0.0000015,
                        "cost_per_token_output": 0.0000020,
                    },
                ],
            },
            {
                "name": "Mistral",
                "class": "llmproxy.provider.huggingface.mistral.Mistral",
                "models": [
                    {
                        "name": "Mistral-7B-v0.1",
                        "cost_per_token_input": 0.00000005,
                        "cost_per_token_output": 0.00000025,
                    },
                ],
            },
            {
                "name": "Cohere",
                "class": "llmproxy.provider.cohere.cohere.Cohere",
                "models": [
                    {
                        "name": "command",
                        "cost_per_token_input": 0.0000015,
                        "cost_per_token_output": 0.000002,
                    },
                ],
            },
        ]
    }

    mock_user_setting = {
        "user_settings": [
            {
                "model": "OpenAI",
                "api_key_var": "OPENAI_API_KEY",
                "max_output_tokens": 256,
                "temperature": 0.1,
                "models": [
                    "gpt-3.5-turbo-1106",
                    "gpt-3.5-turbo-instruct",
                ],
            },
            {
                "model": "Mistral",
                "api_key_var": "MISTRAL_API_KEY",
                "max_output_tokens": 256,
                "temperature": 0.1,
                "models": ["Mistral-7B-v0.1"],
            },
            {
                "model": "Cohere",
                "api_key_var": "COHERE_API_KEY",
                "max_output_tokens": 256,
                "temperature": 0.1,
                "models": ["command"],
            },
        ],
    }
    available_models = _setup_available_models(settings=available_models_setting)

    # Execute the function under test
    user_models = _setup_user_models(
        available_models=available_models, settings=mock_user_setting
    )

    # Assertions
    assert "gpt-3.5-turbo-1106" in user_models
    assert "Mistral-7B-v0.1" in user_models
    assert "command" in user_models

    # settings = _get_settings_from_yml(path_to_yml="llmproxy.config.yml")
    # dev_setting = _get_settings_from_yml(
    #     path_to_yml="llmproxy/config/internal.config.yml"
    # )
    # available_model = _setup_available_models(settings=dev_setting)
    # user_model = _setup_user_models(settings=settings, available_models=available_model)


def test_no_available_model_UserConfigError() -> None:
    text = "Available models not found, please ensure you have the latest version of LLM Proxy."
    with pytest.raises(
        UserConfigError,
        match=text,
    ):
        _setup_user_models(available_models=None, settings=None)

    # test case settings:
    # test_setting = _get_settings_from_yml(path_to_yml="llmproxy/config/internal.config.yml")
    # test_available_model = _setup_available_models(settings=test_setting)
    # test_user_setting = _get_settings_from_yml(path_to_yml="llmproxy.config.yml")
    # test_user_model = _setup_user_models(
    #     available_models=test_available_model, settings=test_user_setting
    # )


def test_setup_user_models_no_setting_UserConfigError():
    with pytest.raises(
        UserConfigError,
        match="Configuration not found, please ensure that you the correct path and format of configuration file",
    ):
        test_setting = _get_settings_from_yml(
            path_to_yml="llmproxy/config/internal.config.yml"
        )
        test_available_model = _setup_available_models(settings=test_setting)
        _setup_user_models(available_models=test_available_model, settings=None)


def test_setup_user_models_empty_user_settings():
    with pytest.raises(
        UserConfigError,
        match="No models found in user settings. Please ensure the format of the configuration file is correct.",
    ):
        test_setting = _get_settings_from_yml(
            path_to_yml="llmproxy/config/internal.config.yml"
        )
        test_available_model = _setup_available_models(settings=test_setting)
        _setup_user_models(
            available_models=test_available_model, settings={"user_settings": []}
        )


def test_setup_user_models_no_variations():
    with pytest.raises(
        LLMProxyConfigError,
        match=r"No models provided in llmproxy\.config\.yml for the following model: OpenAI",
    ):
        _setup_user_models(
            available_models={
                "available_models": [
                    {
                        "name": "OpenAI",
                        "class": "llmproxy.provider.openai.chatgpt.OpenAI",
                        "models": [
                            {
                                "name": "gpt-3.5-turbo-1106",
                            },
                            {
                                "name": "gpt-3.5-turbo-instruct",
                            },
                        ],
                    }
                ]
            },
            settings={"user_settings": [{"model": "OpenAI", "models": None}]},
        )


# TODO: More of an integration test, move later
# TODO: figure out a way for the tests to be routed to the .env.test
def test_cost_routing() -> None:
    # Arrange
    prompt = "I am a man, not a man, but not a man, that is an apple, or a banana!"

    print(CURRENT_DIRECTORY)
    proxy_client = LLMProxy(
        path_to_user_configuration=f"{CURRENT_DIRECTORY}/test.yml",
        path_to_env_vars=".env.test",
    )

    # Act
    output = proxy_client.route(route_type="cost", prompt=prompt)
    # Assert
    # NOTE: You can't do this test; It is not a unit test, it is an integration test
    assert "that is an apple" in output.response


def test_invalid_route_type() -> None:
    prompt = "what's 9+10?"
    with pytest.raises(ValueError, match="'interest' is not a valid RouteType"):
        test = LLMProxy(
            path_to_user_configuration=f"{CURRENT_DIRECTORY}/test.yml",
            path_to_env_vars=".env.test",
        )
        result = test.route(route_type="interest", prompt=prompt)
