import os

import pytest
from dotenv import load_dotenv

from llmproxy.provider.cohere.cohere import Cohere, CohereModel
from llmproxy.utils.exceptions.provider import CohereException, UnsupportedModel

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
    with pytest.raises(CohereException):
        cohere_llm = Cohere(api_key=fake_api_key)
        cohere_llm.get_completion(prompt)


def test_cohere_invalid_model() -> None:
    # Arrange
    cohere_model = "fake model"

    with pytest.raises(UnsupportedModel):
        model = Cohere(model=cohere_model, api_key=cohere_api_key)
        model.get_completion()


def test_cohere_negative_max_token() -> None:
    # Arrange
    num_tokens = -100
    cohere_llm = Cohere(api_key=cohere_api_key, max_output_tokens=num_tokens)
    # Act + Assert
    with pytest.raises(CohereException):
        cohere_llm = Cohere(
            api_key=cohere_api_key,
            max_output_tokens=num_tokens,
        )

        cohere_llm.get_completion()


def test_cohere_negative_temperature() -> None:
    # Arrange
    prompt = "whats 1+1?"
    temperature = -5
    # Act + Assert
    with pytest.raises(CohereException):
        cohere_llm = Cohere(
            prompt=prompt,
            api_key=cohere_api_key,
            max_output_tokens=1000,
            temperature=temperature,
        )

        cohere_llm.get_completion()


def test_cohere_temperature_above_five() -> None:
    # Arrange
    prompt = "whats 1+1?"
    temperature = 6
    # Act + Assert
    with pytest.raises(CohereException):
        cohere_llm = Cohere(
            prompt=prompt,
            api_key=cohere_api_key,
            max_output_tokens=1000,
            temperature=temperature,
        )

        cohere_llm.get_completion()
