import os
from google.cloud import aiplatform
from models.base import BaseChatbot
from utils.enums import BaseEnum
import vertexai
from vertexai.language_models import TextGenerationModel
from dataclasses import dataclass
from dotenv import load_dotenv


load_dotenv()

project_id = os.getenv("GOOGLE_PROJECT_ID")


# class OpenAIModel(str, BaseEnum):
#     GPT_4 = "gpt-4"
#     GPT_4_32K = "gpt-4-32k"
#     GPT_3_5_TURBO = "gpt-3.5-turbo"
#     GPT_3_5_TURBO_16K = "gpt-3.5-turbo-16k"


@dataclass
class VertexAICompletionResponse:
    payload: str = ""
    message: str = ""
    err: str = ""


class VertexAI(BaseChatbot):
    def __init__(
        self,
        prompt: str = "",
        temp: float = 0,
        api_key: str = "",
    ) -> None:
        self

    aiplatform.init(project=project_id, location="us-central1")

    def get_completion(self):
        pass

    def interview(
        temperature: float, project_id: str, location: str, prompt: str
    ) -> str:
        """Ideation example with a Large Language Model"""

        vertexai.init(project=project_id, location=location)
        # TODO developer - override these parameters as needed:
        parameters = {
            "temperature": temperature,  # Temperature controls the degree of randomness in token selection.
            "max_output_tokens": 256,  # Token limit determines the maximum amount of text output.
            "top_p": 0.8,  # Tokens are selected from most probable to least until the sum of their probabilities equals the top_p value.
            "top_k": 40,  # A top_k of 1 means the selected token is the most probable among all tokens.
        }

        model = TextGenerationModel.from_pretrained("text-bison@001")
        response = model.predict(prompt)
        response = f"Response from Model: {response.text}"
        return response

    def getAnswer(prompt: str):
        return interview(0, project_id, "us-central1", prompt)
