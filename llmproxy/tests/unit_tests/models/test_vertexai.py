"""
These tests have multiple warnings about deprecation
They are suppressed and will be addressed later
"""

import os

<<<<<<< HEAD
from llmproxy.models.vertexai import VertexAI, VertexAIException
=======
from llmproxy.provider.google.vertexai import VertexAI
>>>>>>> main
from dotenv import load_dotenv
import unittest

load_dotenv(".env.test")
project_id = os.getenv("GOOGLE_PROJECT_ID")

# This test will only work if you have a valid application_default_credentials.json file


def test_invalid_project_id() -> None:
    # Arrange
    vertexai = VertexAI(project_id="invalid id")

    # Act
    try:
        vertexai.get_completion()
    except VertexAIException as e:
        # Assert
        assert "Permission denied" in str(e)


def test_unsupported_model() -> None:
    # Arrange
    vertexai = VertexAI(project_id=project_id, model="test")

    # Act
    try:
        vertexai.get_completion()
    except VertexAIException as e:
        # Assert
        assert "Model not supported" in str(e)


def test_get_estimated_max_cost():
    # Arrange
    vertex = VertexAI(project_id=project_id)
    expected_cost = 0.000032
    prompt = "I am a cat in a hat!"

    # Act
    actual_cost = vertex.get_estimated_max_cost(prompt=prompt)

    # Assert
    assert (
        actual_cost == expected_cost
    ), "NOTE: Flaky test may need to be changed/removed in future based on pricing"


class TestVertexAIErrors(unittest.TestCase):
    def test_invalid_location(self):
        # Arrange
        vertexai = VertexAI(project_id=project_id, location="test")

        # Act and Assert
        with self.assertRaises(Exception):
            vertexai.get_completion()


if __name__ == "__main__":
    unittest.main()


# test not working properly
# will fix in a different PR
"""
def test_invalid_credentials(monkeypatch) -> None:
    #Arrange
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "TEST")
    vertexai = VertexAI(prompt="What is 1+1?",project_id=project_id)

    #Act
    response = vertexai.get_completion()

    #Assert
    assert "was not found" in response.message
"""
