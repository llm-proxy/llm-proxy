import os
from dotenv import load_dotenv
from llmproxy.models.cohere import Cohere, CohereModel

load_dotenv(".env.test")

#cohere_api_key = os.getenv("COHERE_API_KEY")
cohere_api_key = "iqAm7dJ9TEOwxFSetAfkurVyPjFLwlyQVo0Xd5oe"

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

def test_cohere_empty_api_key() -> None:
    # Arrange
    empty_api_key = ""
    # Act
    cohere_llm = Cohere(api_key=empty_api_key)
    # Assert
    assert(cohere_llm.error_response.err == "ValueError")
    
def test_cohere_invalid_api_key() -> None:
    # Arrange
    fake_api_key = "I am a fake api key"
    prompt = "whats 1+1?"
    # Act
    cohere_llm = Cohere(api_key=fake_api_key)
    response = cohere_llm.get_completion(prompt)
    # Assert
    assert(response.message == "invalid api token")

def test_cohere_invalid_model() -> None:
    # Arrange
    cohere_model = "fake model"
    cohere_llm = Cohere(model=cohere_model)
    # Act
    response = cohere_llm.get_completion()
    # Assert
    assert(response.message == f"Model not supported. Please use one of the following models: {', '.join(CohereModel.list_values())}")

def test_cohere_negative_max_token() -> None:
    # Arrange
    num_tokens = -100
    cohere_llm = Cohere(api_key=cohere_api_key, max_output_tokens=num_tokens)
    # Act
    response = cohere_llm.get_completion()
    # Assert
    assert(response.message == "invalid request: message must be at least 1 token long.")

def test_cohere_negative_temperature() -> None:
    # Arrange
    prompt = "whats 1+1?"
    negative_temp = -1
    cohere_llm = Cohere(prompt=prompt, api_key=cohere_api_key, max_output_tokens=1000, temperature=negative_temp)
    # Act
    response = cohere_llm.get_completion()
    # Assert
    assert(response.message == "invalid request: temperature must be between 0 and 5 inclusive.")

def test_cohere_temperature_above_five() -> None:
    # Arrange
    prompt = "whats 1+1?"
    temperature = 6
    cohere_llm = Cohere(prompt=prompt, api_key=cohere_api_key, max_output_tokens=1000, temperature=temperature)
    # Act
    response = cohere_llm.get_completion()
    # Assert
    assert(response.message == "invalid request: temperature must be between 0 and 5 inclusive.")

def test_cohere_success() -> None:
    # Arrange
    prompt = "whats 1+1?"
    temperature = 0
    cohere_llm = Cohere(prompt=prompt, api_key=cohere_api_key, max_output_tokens=1000, temperature=temperature)
    # Act
    response = cohere_llm.get_completion()
    # Assert
    assert(response.message == "OK")