import os

import pytest
from dotenv import load_dotenv

from proxyllm.provider.anthropic.claude import AnthropicException, ClaudeAdapter

load_dotenv(".env.test")

anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")


def test_empty_api_credentials():
    with pytest.raises(AnthropicException):
        ClaudeAdapter(api_key="", auth_token="").get_completion("Sample prompt")


def test_invalid_api_credentials():
    with pytest.raises(AnthropicException):
        ClaudeAdapter(api_key="invalid_key", auth_token="invalid_token").get_completion(
            "Sample prompt"
        )
