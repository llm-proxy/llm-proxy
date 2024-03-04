import os

import pytest
from dotenv import load_dotenv

from llmproxy.provider.cohere.cohere import CohereAdapter
from llmproxy.utils.exceptions.provider import CohereException, UnsupportedModel

load_dotenv(".env.test")

cohere_api_key = os.getenv("COHERE_API_KEY")


def test_cohere_empty_api_key() -> None:
    # Arrange
    empty_api_key = ""
    # Act + Assert
    with pytest.raises(CohereException):
        co = CohereAdapter(api_key=empty_api_key)
        # This should not cost anything since it errors out before making api request
        co.get_completion()


def test_cohere_invalid_api_key() -> None:
    # Arrange
    fake_api_key = "I am a fake api key"
    prompt = "whats 1+1?"
    with pytest.raises(CohereException):
        cohere_llm = CohereAdapter(api_key=fake_api_key)
        cohere_llm.get_completion(prompt)


# TODO: Slowing down Unit tests, TEST LATER IN INTEGRATION TESTS
def test_mistral_get_estimated_max_cost():
    # Arrange
    cohere = CohereAdapter(api_key=cohere_api_key, max_output_tokens=256)
    expected_cost = 6.44e-05
    prompt = "I am a cat in a hat!"
    price_data = {"prompt": 5e-08, "completion": 2.5e-07}
    # Act
    actual_cost = cohere.get_estimated_max_cost(prompt=prompt, price_data=price_data)

    # Assert
    assert (
        actual_cost == expected_cost
    ), "NOTE: Flaky test may need to be changed/removed in future based on pricing"
