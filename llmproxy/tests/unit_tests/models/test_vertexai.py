"""
These tests have multiple warnings about deprecation
They are suppressed and will be addressed later
"""

import os

from llmproxy.models.vertexai import VertexAI
from dotenv import load_dotenv

load_dotenv(".env.test")
project_id = os.getenv("GOOGLE_PROJECT_ID")

# This test will only work if you have a valid application_default_credentials.json file
def test_invalid_project_id() -> None:
    #Arrange
    vertexai = VertexAI(project_id="invalid id")

    #Act
    response = vertexai.get_completion()

    #Assert
    assert "Permission denied" in response.message

def test_invalid_location() -> None:
    #Arrange
    vertexai = VertexAI(project_id=project_id, location="test")

    #Act
    response = vertexai.get_completion()

    #Assert
    assert "Unsupported region" in response.message

def test_unsupported_model() -> None:
    #Arrange
    vertexai = VertexAI(project_id=project_id,model="test")

    #Act
    response = vertexai.get_completion()

    #Assert
    assert "Model not supported" in response.message

# test not working properly
# will fix in a different PR
'''
def test_invalid_credentials(monkeypatch) -> None:
    #Arrange
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "TEST")
    vertexai = VertexAI(prompt="What is 1+1?",project_id=project_id)

    #Act
    response = vertexai.get_completion()

    #Assert
    assert "was not found" in response.message
'''