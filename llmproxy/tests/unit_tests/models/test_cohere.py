import os
from dotenv import load_dotenv
from llmproxy.models.cohere import Cohere

load_dotenv(".env.test")

cohere_api_key = os.getenv("COHERE_API_KEY")


# TODO: May be a FLAKY test; Ensure this is not the case
def test_get_estimated_max_cost():
    cohere = Cohere(
        api_key=cohere_api_key,
        model="command",
        temperature=0,
    )
    estimated_cost = 0.000112

    prompt = "I am a cat in a hat!"
    actual_cost = cohere.get_estimated_max_cost(prompt=prompt)
    assert actual_cost == estimated_cost, "NOTE: Flaky test may need to be changed/removed in future based on pricing"

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