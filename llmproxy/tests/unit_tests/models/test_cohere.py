import os
from dotenv import load_dotenv
<<<<<<< HEAD
from llmproxy.models.cohere import Cohere, CohereModel
import pytest
=======
from llmproxy.provider.cohere.cohere import Cohere, CohereModel
>>>>>>> main

load_dotenv(".env.test")

cohere_api_key = os.getenv("COHERE_API_KEY")


# TODO: May be a FLAKY test; Ensure this is not the case
def test_get_estimated_max_cost():
    cohere = Cohere(
        api_key=cohere_api_key,
        model="command",
        temperature=0,
    )
    estimated_cost = 0.000112

    prompt = "I am a cat in a hat!"
    actual_cost = cohere.get_estimated_max_cost(prompt=prompt)
    assert (
        actual_cost == estimated_cost
    ), "NOTE: Flaky test may need to be changed/removed in future based on pricing"


def test_cohere_empty_api_key() -> None:
    # Arrange
    empty_api_key = ""
    cohere_llm = None
    error_message = "Cohere Error: No API key provided. Provide the API key in the client initialization or the CO_API_KEY environment variable."
    # Act + Assert
    with pytest.raises(Exception, match=error_message):
        cohere_llm = Cohere(api_key=empty_api_key)


def test_cohere_invalid_api_key() -> None:
    # Arrange
    fake_api_key = "I am a fake api key"
    prompt = "whats 1+1?"
    # Act
    try:
        cohere_llm = Cohere(api_key=fake_api_key)
        response = cohere_llm.get_completion(prompt)
    # Assert
    except:
        assert response.message == "invalid api token"


def test_cohere_invalid_model() -> None:
    # Arrange
    cohere_model = "fake model"
    prompt = "whats 1+1?"
    cohere_llm = None
    # Act
    try:
        cohere_llm = Cohere(model=cohere_model, api_key=cohere_api_key)
    except:
        # Assert
        assert (
            cohere_llm.message
            == f"Model not supported. Please use one of the following models: {', '.join(CohereModel.list_values())}"
        )


def test_cohere_negative_max_token() -> None:
    # Arrange
    num_tokens = -100
    cohere_llm = Cohere(api_key=cohere_api_key, max_output_tokens=num_tokens)
    # Act
    try:
        response = cohere_llm.get_completion()
    # Assert
    except:
        assert (
            response.message
            == "invalid request: message must be at least 1 token long."
        )


def test_cohere_negative_temperature() -> None:
    # Arrange
    prompt = "whats 1+1?"
    negative_temp = -1
    cohere_llm = Cohere(
        prompt=prompt,
        api_key=cohere_api_key,
        max_output_tokens=1000,
        temperature=negative_temp,
    )
    # Act
    try:
        response = cohere_llm.get_completion()
    # Assert
    except:
        assert (
            response.message
            == "invalid request: temperature must be between 0 and 5 inclusive."
        )


def test_cohere_temperature_above_five() -> None:
    # Arrange
    prompt = "whats 1+1?"
    temperature = 6
    cohere_llm = Cohere(
        prompt=prompt,
        api_key=cohere_api_key,
        max_output_tokens=1000,
        temperature=temperature,
    )
    # Act
    try:
        response = cohere_llm.get_completion()
    # Assert
    except:
        assert (
            response.message
            == "invalid request: temperature must be between 0 and 5 inclusive."
        )


def test_cohere_success() -> None:
    # Arrange
    prompt = "whats 1+1?"
    temperature = 0
    cohere_llm = Cohere(
        prompt=prompt,
        api_key=cohere_api_key,
        max_output_tokens=1000,
        temperature=temperature,
    )
    # Act
    try:
        response = cohere_llm.get_completion()
    # Assert
    except:
        assert response.message == "OK"
