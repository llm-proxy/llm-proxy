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
    assert mistral.temp == 1.0


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


def test_mistral_temp_over_100() -> None:
    # Arrange
    mistral = Mistral(api_key=mistral_api_key, temp=100.1)

    # Act
    check = mistral.get_completion()

    print(check.payload)
    # Assert
    assert check.message == "ERROR: Input validation error: `inputs` cannot be empty"


def test_mistral_temp_under_0() -> None:
    # Arrange
    mistral = Mistral(api_key=mistral_api_key, temp=-1)

    # Act
    check = mistral.get_completion()

    print(check.payload)
    # Assert
    assert (
        check.message
        == "ERROR: Input validation error: `temperature` must be strictly positive"
    )
