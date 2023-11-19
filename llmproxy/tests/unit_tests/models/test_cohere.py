import os
from llmproxy.models.cohere import Cohere
from dotenv import load_dotenv

load_dotenv(".env.test")

cohere_api_key = os.getenv("COHERE_API_KEY")

def test_cohere_empty_api_key() -> None:
    # Arrange
    empty_api_key = ""
    # Act
    cohere_llm = Cohere(api_key=empty_api_key)
    # Assert
    assert(cohere_llm.error_response.err == "ValueError")

def test_cohere_invalid_model() -> None:
    # Arrange
    pass
    # Act

    # Assert

def test_cohere_negative_max_token() -> None:
    pass
    # Arrange
    
    # Act

    # Assert

def test_cohere_negative_temperature() -> None:
    pass
    # Arrange
    
    # Act

    # Assert

def test_cohere_temperature_above_five() -> None:
    pass
    # Arrange
    
    # Act

    # Assert