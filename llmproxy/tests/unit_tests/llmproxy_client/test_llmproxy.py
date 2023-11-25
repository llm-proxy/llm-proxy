from llmproxy.llmproxy import (
    _get_settings_from_yml,
    _setup_available_models,
    _setup_user_models,
)
import pytest

import os

from llmproxy.llmproxy import LLMProxy


CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))


# TODO: FIX
def test_empty_model() -> None:
    with pytest.raises(Exception, match="No models provided in user_settings"):
        settings = _get_settings_from_yml(path_to_yml="empty_model_test.yml")
        available_model = _setup_available_models(settings=settings)
        user_model = _setup_user_models(
            settings=settings, available_models=available_model
        )


def test_invalid_model() -> None:
    with pytest.raises(Exception, match="test is not a valid model"):
        settings = _get_settings_from_yml(path_to_yml="invalid_model_test.yml")
        available_model = _setup_available_models(settings=settings)
        user_model = _setup_user_models(
            settings=settings, available_models=available_model
        )


# TODO: ADD TEST
def test_get_settings_from_yml() -> None:
    settings = _get_settings_from_yml(
        path_to_yml=f"{CURRENT_DIRECTORY}/get_setting_test.yml"
    )
    assert settings == "This setting is working"


# TODO: ADD TEST
def test_setup_available_models() -> None:
    settings = _get_settings_from_yml(
        path_to_yml=f"{CURRENT_DIRECTORY}/setup_available_model_test.yml"
    )
    available_model = _setup_available_models(settings=settings)
    print(type(available_model))
    assert type(available_model) is dict


# TODO: ADD TEST
def test_setup_user_models() -> None:
    settings = _get_settings_from_yml(
        path_to_yml=f"{CURRENT_DIRECTORY}/setup_user_setting_test.yml"
    )
    available_model = _setup_available_models(settings=settings)
    user_model = _setup_user_models(settings=settings, available_models=available_model)
    assert type(user_model) is dict


# TODO: More of an integration test, move later
# TODO: figure out a way for the tests to be routed to the .env.test
def test_cost_routing() -> None:
    # Arrange
    prompt = "I am a man, not a man, but not a man, that is an apple, or a banana!"

    proxy_client = LLMProxy(
        path_to_configuration=f"{CURRENT_DIRECTORY}/test.yml",
        path_to_env_vars=".env.test",
    )

    # Act
    output = proxy_client.route(route_type="cost", prompt=prompt)

    # Assert
    assert output.payload
    assert not output.err
