from google.cloud import aiplatform
from google.auth import exceptions
from llmproxy.models.base import BaseChatbot, CompletionResponse
from llmproxy.utils.enums import BaseEnum
from llmproxy.utils.log import logger
from vertexai.language_models import TextGenerationModel

class VertexAIModel(str, BaseEnum):
    PALM_TEXT = "text-bison@001"
    PALM_CHAT = "chat-bison"

class VertexAI(BaseChatbot):
    def __init__(
        self,
        prompt: str = "",
        temperature: float = 0,
        model: VertexAIModel = VertexAIModel.PALM_TEXT.value,
        project_id: str | None = "",
        location: str | None = "",
    ) -> None:
        self.prompt = prompt
        self.temperature = temperature
        self.model = model
        aiplatform.init(project=project_id, location=location)

    def get_completion(self) -> CompletionResponse:
        if self.model not in VertexAIModel:
            return self._handle_error(
                exception = f"Model not supported Please use one of the following models: {', '.join(VertexAIModel.list_values())}",
                error_type = "ValueError"
            )
        try:
            # TODO developer - override these parameters as needed:
            parameters = {
                "temperature": self.temperature,  # Temperature controls the degree of randomness in token selection.
                "max_output_tokens": 256,  # Token limit determines the maximum amount of text output.
                "top_p": 0.8,  # Tokens are selected from most probable to least until the sum of their probabilities equals the top_p value.
                "top_k": 40,  # A top_k of 1 means the selected token is the most probable among all tokens.
            }

            chat_model = TextGenerationModel.from_pretrained(self.model)
            response = chat_model.predict(self.prompt)
            output = response.text

        except exceptions.GoogleAuthError as e:
            logger.error(e.args[0])
            return CompletionResponse(message=e.args[0], err=type(e).__name__)
        except Exception as e:
            logger.error(e.args[0])
            raise Exception(e)

        return CompletionResponse(payload=output, message="OK", err="")
    
    def _handle_error(self, exception: str, error_type: str) -> CompletionResponse:
        return CompletionResponse(message=exception, err=error_type)