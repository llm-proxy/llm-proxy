import os

import pytest
from dotenv import load_dotenv

from llmproxy.provider.openai.chatgpt import OpenAIAdapter, OpenAIException
from llmproxy.utils.exceptions.provider import UnsupportedModel

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


def test_unsupported_model():
    chatbot = OpenAIAdapter(api_key=openai_api_key, model="unsupported_model")

    with pytest.raises(UnsupportedModel):
        chatbot.get_completion()


# TODO: Slowing down Unit tests too much, TEST LATER IN INTEGRATION TESTS
def test_get_estimated_max_cost():
    # Arrange
    gpt = OpenAIAdapter(api_key=openai_api_key)
    expected_cost = 0.000108
    prompt = "I am a cat in a hat!"

    # Act
    actual_cost = gpt.get_estimated_max_cost(prompt=prompt)

    # Assert
    assert (
        actual_cost == expected_cost
    ), "NOTE: Flaky test may need to be changed/removed in future based on pricing"
