"""
These tests have multiple warnings about deprecation
They are suppressed and will be addressed later
"""

import os

import pytest
from dotenv import load_dotenv

from proxyllm.provider.google.vertexai import VertexAIAdapter, VertexAIException

load_dotenv(".env.test")
project_id = os.getenv("GOOGLE_PROJECT_ID")

# This test will only work if you have a valid application_default_credentials.json file


def test_invalid_project_id() -> None:
    # Arrange
    vertexai = VertexAIAdapter(project_id="invalid id")

    # Act + Assert
    with pytest.raises(VertexAIException):
        vertexai.get_completion()


def test_invalid_location():
    # Assert
    with pytest.raises(VertexAIException):
        # Arrange
        vertexai = VertexAIAdapter(project_id=project_id, location="test")
        # Act
        vertexai.get_completion()


def test_vertexai_tokenize_returns_expected_num_of_input_tokens():
    # Arrange
    expected_num_of_input_tokens = 36
    max_output_tokens = 256
    vertexai = VertexAIAdapter(max_output_tokens=100)
    prompt = "The quick brown fox jumps over the lazy dog."

    # Act
    encoding = vertexai.tokenize(prompt=prompt)

    # Assert
    assert (
        encoding.num_of_input_tokens == expected_num_of_input_tokens
    ), "NOTE: Flaky test may need to be changed/removed in future based on pricing"
