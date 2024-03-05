import os

import pytest
from dotenv import load_dotenv

from proxyllm.provider.huggingface.mistral import MistralAdapter, MistralException

load_dotenv(".env.test")

mistral_api_key = os.getenv("HUGGING_FACE_API_KEY")


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


# TODO: Slowing down Unit tests too much, TEST LATER IN INTEGRATION TESTS
def test_mistral_get_estimated_max_cost():
    # Arrange
    mistral = MistralAdapter(api_key=mistral_api_key, max_output_tokens=256)
    expected_cost = 6.44e-05
    prompt = "I am a cat in a hat!"
    price_data = {"prompt": 5e-08, "completion": 2.5e-07}
    # Act
    actual_cost = mistral.get_estimated_max_cost(prompt=prompt, price_data=price_data)

    # Assert
    assert (
        actual_cost == expected_cost
    ), "NOTE: Flaky test may need to be changed/removed in future based on pricing"
