import os

from llmproxy.models.mistral import Mistral
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
    # Arrange
    mistral = Mistral(api_key="invalid")

    # Act
    check = mistral.get_completion()

    # Assert
    assert (
        check.message
        == "ERROR: Authorization header is correct, but the token seems invalid"
    )


def test_mistral_temperature_over_100() -> None:
    # Arrange
    mistral = Mistral(api_key=mistral_api_key, temperature=100.1)

    # Act
    check = mistral.get_completion()

    print(check.payload)
    # Assert
    assert check.message == "ERROR: Input validation error: `inputs` cannot be empty"


def test_mistral_temperature_under_0() -> None:
    # Arrange
    mistral = Mistral(api_key=mistral_api_key, temperature=-1)

    # Act
    check = mistral.get_completion()

    print(check.payload)
    # Assert
    assert (
        check.message
        == "ERROR: Input validation error: `temperature` must be strictly positive"
    )


def test_get_estimated_max_cost():
    # Arrange
    mistral = Mistral(
        api_key=mistral_api_key,
    )
    prompt = "I am a cat in a hat!"
    estimated_cost = 0.0000954

    # Act
    actual_cost = mistral.get_estimated_max_cost(prompt=prompt)
    assert (
        actual_cost == estimated_cost
    ), "NOTE: Flaky test may need to be changed/removed in future based on pricing"
