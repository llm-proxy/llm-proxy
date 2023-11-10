from google.cloud import aiplatform
from google.auth import exceptions as auth_exceptions
from google.api_core import exceptions as api_exceptions
from llmproxy.models.base import BaseModel, CompletionResponse
from llmproxy.utils.enums import BaseEnum
from llmproxy.utils.log import logger
from vertexai.language_models import TextGenerationModel


class VertexAIModel(str, BaseEnum):
    PALM_TEXT = "text-bison@001"
    PALM_CHAT = "chat-bison"


class VertexAI(BaseModel):
    def __init__(
        self,
        prompt: str = "",
        temperature: float = 0,
        model: VertexAIModel = VertexAIModel.PALM_TEXT.value,
        project_id: str | None = "",
        location: str | None = "",
        max_output_tokens: int = None,
    ) -> None:
        self.prompt = prompt
        self.temperature = temperature
        self.model = model
        self.project_id = project_id
        self.location = location
        self.max_output_tokens = max_output_tokens

    def get_completion(self, prompt: str = "") -> CompletionResponse:
        if self.model not in VertexAIModel:
            return self._handle_error(
                exception=f"Model not supported Please use one of the following models: {', '.join(VertexAIModel.list_values())}",
                error_type="ValueError",
            )
        try:
            aiplatform.init(project=self.project_id, location=self.location)
            # TODO developer - override these parameters as needed:
            parameters = {
                # Temperature controls the degree of randomness in token selection.
                "temperature": self.temperature,
                # Token limit determines the maximum amount of text output.
                "max_output_tokens": self.max_output_tokens,
            }

            chat_model = TextGenerationModel.from_pretrained(self.model)
            response = chat_model.predict(prompt if prompt else self.prompt)
            output = response.text

        except api_exceptions.GoogleAPIError as e:
            logger.error(e.args[0])
            return self._handle_error(exception=e.args[0], error_type=type(e).__name__)
        except auth_exceptions.GoogleAuthError as e:
            logger.error(e.args[0])
            return self._handle_error(exception=e.args[0], error_type=type(e).__name__)
        except ValueError as e:
            logger.error(e.args[0])
            return self._handle_error(exception=e.args[0], error_type=type(e).__name__)
        except Exception as e:
            logger.error(e.args[0])
            raise Exception(e)

        return CompletionResponse(payload=output, message="OK", err="")

    def _handle_error(self, exception: str, error_type: str) -> CompletionResponse:
        return CompletionResponse(message=exception, err=error_type)
