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
    pass


# TODO: ADD TEST
def test_setup_available_models() -> None:
    pass


# TODO: ADD TEST
def test_setup_user_models() -> None:
    pass


# TODO: More of an integration test, move later
def test_cost_routing() -> None:
    # Arrange
    prompt = "I am a man, not a man, but not a man, that is an apple, or a banana!"

    proxy_client = LLMProxy(path_to_configuration=f"{CURRENT_DIRECTORY}/test.yml")

    # Act
    output = proxy_client.route(route_type="cost", prompt=prompt)

    # Assert
    assert output.payload
    assert not output.err
