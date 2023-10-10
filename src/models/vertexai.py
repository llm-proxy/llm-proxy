from google.cloud import aiplatform
from google.auth import exceptions
from models.base import BaseChatbot, CompletionResponse
from utils.enums import BaseEnum
import vertexai
from vertexai.language_models import ChatModel


class VertexAIModel(str, BaseEnum):
    CHAT_BISON = "chat-bison@001"


class VertexAI(BaseChatbot):
    def __init__(
        self,
        prompt: str = "",
        temperature: float = 0,
        model: VertexAIModel = VertexAIModel.CHAT_BISON,
        project_id: str | None = "",
        location: str = "",
    ) -> None:
        self.prompt = prompt
        self.temperature = temperature
        self.model = model
        aiplatform.init(project=project_id, location=location)
        vertexai.init(project=project_id, location=location)

    def get_completion(self) -> CompletionResponse:
        try:
            # TODO developer - override these parameters as needed:
            parameters = {
                "temperature": self.temperature,  # Temperature controls the degree of randomness in token selection.
                "max_output_tokens": 256,  # Token limit determines the maximum amount of text output.
                "top_p": 0.8,  # Tokens are selected from most probable to least until the sum of their probabilities equals the top_p value.
                "top_k": 40,  # A top_k of 1 means the selected token is the most probable among all tokens.
            }

            chat_model = ChatModel.from_pretrained(self.model)
            chat = chat_model.start_chat()
            response = chat.send_message("What is 1+1", **parameters)
            output = f"Response from Model: {response.text}"
        except exceptions.GoogleAuthError as e:
            return CompletionResponse(message=e.args[0], err=type(e).__name__)
        except Exception as e:
            raise Exception(e)

        return CompletionResponse(payload=output, message="OK", err="")
