from google.cloud import aiplatform
from models.base import BaseChatbot, CompletionResponse
from utils.enums import BaseEnum
import vertexai
from vertexai.language_models import TextGenerationModel
from dataclasses import dataclass


class VertexAI(BaseChatbot):
    def __init__(
        self,
        prompt: str = "",
        temperature: float = 0,
        project_id: str = "",
        location: str = "",
    ) -> None:
        self.prompt = prompt
        self.temperature = temperature
        aiplatform.init(project=project_id, location=location)
        vertexai.init(project=project_id, location=location)

    def get_completion(self) -> CompletionResponse:
        # TODO developer - override these parameters as needed:
        parameters = {
            "temperature": self.temperature,  # Temperature controls the degree of randomness in token selection.
            "max_output_tokens": 256,  # Token limit determines the maximum amount of text output.
            "top_p": 0.8,  # Tokens are selected from most probable to least until the sum of their probabilities equals the top_p value.
            "top_k": 40,  # A top_k of 1 means the selected token is the most probable among all tokens.
        }
        model = TextGenerationModel.from_pretrained("text-bison@001")

        response = model.predict(prompt)
        response = f"Response from Model: {response.text}"
        return response

        pass

    def interview(
        temperature: float, project_id: str, location: str, prompt: str
    ) -> str:
        """Ideation example with a Large Language Model"""

    def getAnswer(prompt: str):
        return interview(0, project_id, "us-central1", prompt)
