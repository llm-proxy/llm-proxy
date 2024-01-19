import os

from unittest.mock import patch
from openai import error
from dotenv import load_dotenv
import pytest
from llmproxy.utils.exceptions.provider import UnsupportedModel
from llmproxy.provider.openai.chatgpt import OpenAI, OpenAIException


load_dotenv(".env.test")

openai_api_key = os.getenv("OPENAI_API_KEY")


def test_invalid_api_key():
    # Assert
    with pytest.raises(OpenAIException):
        # Arrange
        chatbot = OpenAI(api_key="invalid_key")
        # Act
        chatbot.get_completion()


def test_unsupported_model():
    chatbot = OpenAI(api_key=openai_api_key, model="unsupported_model")

    with pytest.raises(UnsupportedModel):
        chatbot.get_completion()


def test_generic_exception():
    with patch("openai.ChatCompletion.create", side_effect=Exception("Random error")):
        chatbot = OpenAI(api_key=openai_api_key)
        try:
            chatbot.get_completion()
        except Exception as e:
            assert str(e) == "OpenAI Error: Random error, Type: Unknown OpenAI Error"


def test_openai_rate_limit_error():
    with patch(
        "openai.ChatCompletion.create",
        side_effect=error.OpenAIError("Rate limit exceeded"),
    ):
        chatbot = OpenAI(api_key=openai_api_key)
        try:
            chatbot.get_completion()
        except OpenAIException as e:
            assert str(e) == "OpenAI Error: Rate limit exceeded, Type: OpenAIError"


def test_get_estimated_max_cost():
    # Arrange
    gpt = OpenAI(api_key=openai_api_key)
    expected_cost = 0.000108
    prompt = "I am a cat in a hat!"

    # Act
    actual_cost = gpt.get_estimated_max_cost(prompt=prompt)

    # Assert
    assert (
        actual_cost == expected_cost
    ), "NOTE: Flaky test may need to be changed/removed in future based on pricing"
