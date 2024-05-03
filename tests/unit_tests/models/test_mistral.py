import os

import pytest
from dotenv import load_dotenv

from proxyllm.provider.mistral.mistral import MistralAdapter, MistralException

load_dotenv(".env.test")

mistral_api_key = os.getenv("MISTRAL_API_KEY")


def test_mistral_constructor_default() -> None:
    # Arrange
    mistral = MistralAdapter()

    # Act

    # Assert
    assert mistral.api_key == ""
    assert mistral.prompt == ""
    assert mistral.temperature == 1.0


def test_mistral_invalid_api_key() -> None:
    # Assert
    with pytest.raises(MistralException):
        # Arrange
        mistral = MistralAdapter(api_key="invalid")
        # Act
        mistral.get_completion()


def test_mistral_temperature_under_0() -> None:
    # Assert
    with pytest.raises(MistralException):
        # Arrange
        mistral = MistralAdapter(api_key=mistral_api_key, temperature=-1)

        # Act
        mistral.get_completion()


def test_mistral_tokenize_returns_expected_num_of_input_tokens():
    # Arrange
    expected_num_of_input_tokens = 11
    max_output_tokens = 256
    mistral = MistralAdapter(api_key=mistral_api_key, max_output_tokens=100)
    prompt = "The quick brown fox jumps over the lazy dog."

    # Act
    encoding = mistral.tokenize(prompt=prompt)

    # Assert
    assert (
        encoding.num_of_input_tokens == expected_num_of_input_tokens
    ), "NOTE: Flaky test may need to be changed/removed in future based on pricing"
