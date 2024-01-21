import os

import pytest
from dotenv import load_dotenv

from llmproxy.provider.huggingface.llama2 import Llama2, Llama2Exception, Llama2Model
from llmproxy.utils.exceptions.provider import EmptyPrompt, UnsupportedModel

load_dotenv(".env.test")
"""This test assumes a free version access token from huggingface"""
default_api_key = os.getenv("LLAMA2_API_KEY")

default_prompt = "What's 1+1?"
default_system_prompt = "Answer correctly"
default_model = Llama2Model.LLAMA_2_7B.value


def test_llama2_empty_prompt() -> None:
    llama2_empty_prompt = Llama2(
        prompt="",
        system_prompt=default_system_prompt,
        model=default_model,
        api_key=default_api_key,
    )

    with pytest.raises(EmptyPrompt):
        llama2_empty_prompt.get_completion()


def test_llama2_invalid_api_key() -> None:
    test_api_key = Llama2(
        prompt=default_prompt,
        system_prompt=default_system_prompt,
        api_key="LMAO-key",
        model=default_model,
    )

    with pytest.raises(Llama2Exception):
        test_api_key.get_completion()


def test_llama2_free_subscription_api_key() -> None:
    test_model = Llama2(
        prompt=default_prompt,
        system_prompt=default_system_prompt,
        api_key=default_api_key,
        model=Llama2Model.LLAMA_2_7B.value,
    )

    with pytest.raises(Llama2Exception):
        test_model.get_completion()


def test_llama2_emp_model() -> None:
    test_emp_model = Llama2(
        prompt=default_prompt,
        system_prompt=default_system_prompt,
        api_key=default_api_key,
        model="",
    )

    with pytest.raises(UnsupportedModel):
        test_emp_model.get_completion()
