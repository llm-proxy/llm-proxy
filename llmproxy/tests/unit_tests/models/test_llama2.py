import os

import pytest
from dotenv import load_dotenv

from llmproxy.provider.huggingface.llama2 import Llama2Adapter
from llmproxy.utils.exceptions.provider import EmptyPrompt, Llama2Exception

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
