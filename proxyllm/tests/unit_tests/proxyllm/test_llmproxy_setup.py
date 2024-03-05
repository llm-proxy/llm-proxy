import os

import pytest

from proxyllm.config.internal_config import internal_config
from proxyllm.proxyllm import (
    UserConfigError,
    _setup_available_models,
    _setup_user_models,
)

CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
PATH_TO_ENV_TEST = ".env.test"


def test_setup_available_models() -> None:
    _setup_available_models(settings=internal_config)


def test_no_available_model_UserConfigError() -> None:
    text = "Available models not found, please ensure you have the latest version of LLM Proxy."
    with pytest.raises(
        UserConfigError,
        match=text,
    ):
        _setup_user_models(available_models={}, yml_settings={})


def test_setup_user_models_no_setting_UserConfigError():
    with pytest.raises(
        UserConfigError,
        match="Configuration not found, please ensure that you the correct path and format of configuration file",
    ):
        test_available_model = _setup_available_models(settings=internal_config)
        _setup_user_models(available_models=test_available_model, yml_settings={})


def test_setup_user_models_empty_user_settings():
    with pytest.raises(
        UserConfigError,
        match="No models found in user settings. Please ensure the format of the configuration file is correct.",
    ):
        test_available_model = _setup_available_models(settings=internal_config)
        _setup_user_models(
            available_models=test_available_model,
            yml_settings={"provider_settings": []},
        )


def test_setup_user_models_no_variation() -> None:
    with pytest.raises(
        UserConfigError,
    ):
        test_available_model = _setup_available_models(settings=internal_config)

        _setup_user_models(
            available_models=test_available_model,
            yml_settings={
                "provider_settings": [
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
