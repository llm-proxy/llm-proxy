import os

import pytest
from dotenv import load_dotenv

from llmproxy.provider.huggingface.mistral import Mistral, MistralException

load_dotenv(".env.test")

mistral_api_key = os.getenv("MISTRAL_API_KEY")


def test_mistral_constructor_default() -> None:
    # Arrange
    mistral = Mistral()

    # Act

    # Assert
    assert mistral.api_key == ""
    assert mistral.prompt == ""
    assert mistral.temperature == 1.0


def test_mistral_invalid_api_key() -> None:
    # Assert
    with pytest.raises(MistralException):
        # Arrange
        mistral = Mistral(api_key="invalid")
        # Act
        mistral.get_completion()


def test_mistral_temperature_under_0() -> None:
    # Assert
    with pytest.raises(MistralException):
        # Arrange
        mistral = Mistral(api_key=mistral_api_key, temperature=-1)

        # Act
        mistral.get_completion()
