from llmproxy.llmproxy import (
    _get_settings_from_yml,
    _setup_available_models,
    _setup_user_models,
)
import pytest
import os
from dotenv import load_dotenv

load_dotenv(".env.test")


def empty_model_test() -> None:
    with pytest.raises(Exception, match="No models provided in user_settings"):
        settings = _get_settings_from_yml(path_to_yml="empty_model_test.yml")
        available_model = _setup_available_models(settings=settings)
        user_model = _setup_user_models(
            settings=settings, available_models=available_model
        )


def invalid_model_test() -> None:
    with pytest.raises(Exception, match="test is not a valid model"):
        settings = _get_settings_from_yml(path_to_yml="invalid_model_test.yml")
        available_model = _setup_available_models(settings=settings)
        user_model = _setup_user_models(
            settings=settings, available_models=available_model
        )
