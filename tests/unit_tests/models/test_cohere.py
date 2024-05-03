import os

import pytest
from dotenv import load_dotenv

from proxyllm.provider.cohere.cohere import CohereAdapter, CohereException

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


def test_cohere_tokenize_returns_expected_num_of_input_tokens():
    # Arrange
    expected_num_of_input_tokens = 11
    max_output_tokens = 256
    cohere = CohereAdapter(api_key=cohere_api_key, max_output_tokens=100)
    prompt = "The quick brown fox jumps over the lazy dog."

    # Act
    encoding = cohere.tokenize(prompt=prompt)

    # Assert
    assert (
        encoding.num_of_input_tokens == expected_num_of_input_tokens
    ), "NOTE: Flaky test may need to be changed/removed in future based on pricing"
