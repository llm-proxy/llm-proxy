import os

import pytest
from dotenv import load_dotenv

from proxyllm.provider.openai.chatgpt import OpenAIAdapter, OpenAIException
from proxyllm.utils.exceptions.provider import UnsupportedModel

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


# TODO: Slowing down Unit tests too much, TEST LATER IN INTEGRATION TESTS
def test_get_estimated_max_cost():
    # Arrange
    gpt = OpenAIAdapter(
        api_key=openai_api_key, max_output_tokens=256, model="gpt-3.5-turbo-1106"
    )
    expected_cost = 0.00052
    prompt = "I am a cat in a hat!"
    price_data = {"prompt": 1e-06, "completion": 2e-06}
    # Act
    actual_cost = gpt.get_estimated_max_cost(prompt=prompt, price_data=price_data)

    # Assert
    assert (
        actual_cost == expected_cost
    ), "NOTE: Flaky test may need to be changed/removed in future based on pricing"
