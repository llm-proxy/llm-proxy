import os

import pytest
from dotenv import load_dotenv

from proxyllm.provider.anthropic.claude import AnthropicException, ClaudeAdapter

load_dotenv(".env.test")

anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")


def test_empty_api_key():
    with pytest.raises(AnthropicException):
        adapter = ClaudeAdapter(api_key="")
        adapter.get_completion()


def test_invalid_api_credentials():
    with pytest.raises(AnthropicException):
        chatbot = ClaudeAdapter(api_key="invalid_key")
        chatbot.get_completion()


def test_claude_tokenize_returns_expected_num_of_input_tokens():
    # Set up your expected number of tokens based on the 'prompt' string
    expected_num_of_input_tokens = 10
    max_output_tokens = 256
    claude = ClaudeAdapter(
        api_key=anthropic_api_key,
        max_output_tokens=max_output_tokens,
        model="claude-3-opus-20240229",
    )
    prompt = "The quick brown fox jumps over the lazy dog."
    tokenize_response = claude.tokenize(prompt=prompt)

    assert (
        tokenize_response.num_of_input_tokens == expected_num_of_input_tokens
    ), "The number of input tokens does not match the expected value"
