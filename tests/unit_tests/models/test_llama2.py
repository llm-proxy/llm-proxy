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


def test_llama2_tokenize_returns_expected_num_of_input_tokens():
    # Arrange
    expected_num_of_input_tokens = 11
    max_output_tokens = 256
    llama2 = Llama2Adapter(api_key=default_api_key, max_output_tokens=100)
    prompt = "The quick brown fox jumps over the lazy dog."

    # Act
    encoding = llama2.tokenize(prompt=prompt)

    # Assert
    assert (
        encoding.num_of_input_tokens == expected_num_of_input_tokens
    ), "NOTE: Flaky test may need to be changed/removed in future based on pricing"
