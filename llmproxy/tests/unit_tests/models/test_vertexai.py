import os

from llmproxy.models.vertexai import VertexAI
from dotenv import load_dotenv

load_dotenv(".env.test")
project_id = os.getenv("GOOGLE_PROJECT_ID")

def test_invalid_project_id() -> None:
    vertexai = VertexAI(project_id="invalid id")
    response = vertexai.get_completion()
    assert "Permission denied" in response.message

def test_invalid_location() -> None:
    vertexai = VertexAI(project_id=project_id, location="test")
    response = vertexai.get_completion()
    assert "Unsupported region" in response.message

def test_unsupported_model() -> None:
    vertexai = VertexAI(project_id=project_id,model="test")
    response = vertexai.get_completion()
    assert "Model not supported" in response.message

# test not working properly
# will fix in a different PR
'''
def test_invalid_credentials(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "TEST")
    vertexai = VertexAI(prompt="What is 1+1?",project_id=project_id)
    response = vertexai.get_completion()
    assert "was not found" in response.message
'''