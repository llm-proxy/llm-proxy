import os

import pytest

from llmproxy.provider.huggingface.mistral import Mistral, MistralException
from dotenv import load_dotenv

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


# test no longer works due to updated pricing, will update in the future
# def test_get_estimated_max_cost():
#     # Arrange
#     mistral = Mistral(
#         api_key=mistral_api_key,
#     )
#     prompt = "I am a cat in a hat!"
#     estimated_cost = 0.0000954

#     # Act
#     actual_cost = mistral.get_estimated_max_cost(prompt=prompt)
#     assert (
#         actual_cost == estimated_cost
#     ), "NOTE: Flaky test may need to be changed/removed in future based on pricing"
