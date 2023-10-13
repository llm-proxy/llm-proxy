import os
from llmproxy.models.openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

def test_invalid_temperature():
    chatbot = OpenAI(api_key = openai_api_key, temp=-1)
    response = chatbot.get_completion()
    assert isinstance(response.err, str)
    assert "temperature" in response.message.lower()

def test_invalid_api_key():
    chatbot = OpenAI(api_key="invalid_key")
    response = chatbot.get_completion()
    assert isinstance(response.err, str)
    assert "authentication" in response.message.lower()

def test_unsupported_model():
    chatbot = OpenAI(api_key=openai_api_key, model="unsupported_model")
    response = chatbot.get_completion()
    assert response.err == "ValueError"
    assert response.message == "Model not supported"




