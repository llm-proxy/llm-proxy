import os

import pytest
from dotenv import load_dotenv

from proxyllm.provider.huggingface.llama2 import Llama2Adapter
from proxyllm.utils.exceptions.provider import EmptyPrompt, Llama2Exception

load_dotenv(".env.test")
"""This test assumes a free version access token from huggingface"""
default_api_key = os.getenv("HUGGING_FACE_API_KEY")

default_prompt = "What's 1+1?"
default_system_prompt = "Answer correctly"
default_model = "Llama-2-7b-chat-hf"


def test_llama2_empty_prompt() -> None:
    llama2_empty_prompt = Llama2Adapter(
        prompt="",
        system_prompt=default_system_prompt,
        model=default_model,
        api_key=default_api_key,
    )

    with pytest.raises(EmptyPrompt):
        llama2_empty_prompt.get_completion()


def test_llama2_invalid_api_key() -> None:
    test_api_key = Llama2Adapter(
        prompt=default_prompt,
        system_prompt=default_system_prompt,
        api_key="LMAO-key",
        model=default_model,
    )

    with pytest.raises(Llama2Exception):
        test_api_key.get_completion()


def test_llama2_free_subscription_api_key() -> None:
    test_model = Llama2Adapter(
        prompt=default_prompt,
        system_prompt=default_system_prompt,
        api_key=default_api_key,
        model=default_model,
    )

    with pytest.raises(Llama2Exception):
        test_model.get_completion()


# TODO: Slowing down Unit tests, TEST LATER IN INTEGRATION TESTS
def test_llama2_get_estimated_max_cost():
    # Arrange
    llama2 = Llama2Adapter(api_key=default_api_key, max_output_tokens=256)
    expected_cost = 6.44e-05
    prompt = "I am a cat in a hat!"
    price_data = {"prompt": 5e-08, "completion": 2.5e-07}
    # Act
    actual_cost = llama2.get_estimated_max_cost(
        prompt=prompt, price_data=price_data
    ).cost

    # Assert
    assert (
        actual_cost == expected_cost
    ), "NOTE: Flaky test may need to be changed/removed in future based on pricing"
