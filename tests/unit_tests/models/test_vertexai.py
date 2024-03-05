"""
These tests have multiple warnings about deprecation
They are suppressed and will be addressed later
"""

import os
import unittest

import pytest
from dotenv import load_dotenv

from proxyllm.provider.google.vertexai import VertexAIAdapter, VertexAIException
from proxyllm.utils.exceptions.provider import UnsupportedModel

load_dotenv(".env.test")
project_id = os.getenv("GOOGLE_PROJECT_ID")

# This test will only work if you have a valid application_default_credentials.json file


def test_invalid_project_id() -> None:
    # Arrange
    vertexai = VertexAIAdapter(project_id="invalid id")

    # Act + Assert
    with pytest.raises(VertexAIException):
        vertexai.get_completion()


def test_get_estimated_max_cost():
    # Arrange
    vertex = VertexAIAdapter(project_id=project_id, max_output_tokens=256)
    expected_cost = 0.000533
    prompt = "I am a cat in a hat!"
    price_data = {"prompt": 1.5e-06, "completion": 2e-06}

    # Act
    actual_cost = vertex.get_estimated_max_cost(prompt=prompt, price_data=price_data)

    # Assert
    assert (
        actual_cost == expected_cost
    ), "NOTE: Flaky test may need to be changed/removed in future based on pricing"


def test_invalid_location():
    # Assert
    with pytest.raises(VertexAIException):
        # Arrange
        vertexai = VertexAIAdapter(project_id=project_id, location="test")
        # Act
        vertexai.get_completion()


if __name__ == "__main__":
    unittest.main()
