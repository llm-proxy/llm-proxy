from llmproxy.llmproxy import (
    _get_settings_from_yml,
    _setup_available_models,
    _setup_user_models,
)
import pytest

import os

from llmproxy.llmproxy import LLMProxy


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
            path_to_env_vars=".env.test"
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
        settings = _get_settings_from_yml(
            path_to_yml=f"{CURRENT_DIRECTORY}/invalid_model_test.yml"
        )
        available_model = _setup_available_models(settings=settings)
        user_model = _setup_user_models(
            settings=settings, available_models=available_model
        )


# TODO: ADD TEST
def test_get_settings_from_yml() -> None:
    user_setting = _get_settings_from_yml(path_to_yml="llmproxy.config.yml")
    setting = _get_settings_from_yml(path_to_yml="llmproxy/config/internal.config.yml")


# TODO: ADD TEST
def test_setup_available_models() -> None:
    settings = _get_settings_from_yml(path_to_yml="llmproxy/config/internal.config.yml")
    available_model = _setup_available_models(settings=settings)


# TODO: ADD TEST
def test_setup_user_models() -> None:
    settings = _get_settings_from_yml(path_to_yml="llmproxy.config.yml")
    dev_setting = _get_settings_from_yml(
        path_to_yml="llmproxy/config/internal.config.yml"
    )
    available_model = _setup_available_models(settings=dev_setting)
    user_model = _setup_user_models(settings=settings, available_models=available_model)


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
