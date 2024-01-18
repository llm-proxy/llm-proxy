import os

from dotenv import load_dotenv
<<<<<<< HEAD
from llmproxy.models.openai import OpenAI, OpenAIException
=======
from llmproxy.provider.openai.chatgpt import OpenAI
>>>>>>> main
from openai import error
from unittest.mock import patch

load_dotenv(".env.test")

openai_api_key = os.getenv("OPENAI_API_KEY")


def test_invalid_api_key():
    chatbot = OpenAI(api_key="invalid_key")
    try:
        chatbot.get_completion()
    except OpenAIException as e:
        assert (
            str(e)
            == "OpenAI Error: Incorrect API key provided: invalid_key. You can find your API key at https://platform.openai.com/account/api-keys., Type: AuthenticationError"
        )


def test_unsupported_model():
    chatbot = OpenAI(api_key=openai_api_key, model="unsupported_model")
    try:
        chatbot.get_completion()
    except OpenAIException as e:
        assert "Model not supported" in str(e)


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
