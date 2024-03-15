import os

import pytest
from dotenv import load_dotenv

from proxyllm.provider.openai.chatgpt import OpenAIAdapter, OpenAIException

load_dotenv(".env.test")

openai_api_key = os.getenv("OPENAI_API_KEY")


def test_empty_api_key():
    # Assert
    with pytest.raises(OpenAIException):
        # Arrange
        chatbot = OpenAIAdapter(api_key="")
        # Act
        chatbot.get_completion()


def test_invalid_api_key():
    # Assert
    with pytest.raises(OpenAIException):
        # Arrange
        chatbot = OpenAIAdapter(api_key="invalid_key")
        # Act
        chatbot.get_completion()


def test_openai_tokenize_returns_expected_num_of_input_tokens():
    # Arrange
    expected_num_of_input_tokens = 10
    max_output_tokens = 256
    openai = OpenAIAdapter(api_key=openai_api_key, max_output_tokens=100, model="gpt-4")
    prompt = "The quick brown fox jumps over the lazy dog."

    # Act
    encoding = openai.tokenize(prompt=prompt)

    # Assert
    assert (
        encoding.num_of_input_tokens == expected_num_of_input_tokens
    ), "NOTE: Flaky test may need to be changed/removed in future based on pricing"
