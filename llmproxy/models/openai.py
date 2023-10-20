from llmproxy.models.base import BaseChatbot, CompletionResponse
from llmproxy.utils.enums import BaseEnum
from llmproxy.utils.log import logger
import openai
from openai import error


class OpenAIModel(str, BaseEnum):
    GPT_4 = "gpt-4"
    GPT_4_32K = "gpt-4-32k"
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_3_5_TURBO_16K = "gpt-3.5-turbo-16k"


class OpenAI(BaseChatbot):
    def __init__(
        self,
        prompt: str = "",
        model: OpenAIModel = OpenAIModel.GPT_3_5_TURBO.value,
        temp: float = 0,
        api_key: str = "",
    ) -> None:
        self.prompt = prompt
        self.model = model
        self.temp = temp
        # We may have to pull this directly from .env and use different .env file/names for testing
        openai.api_key = api_key

    def get_completion(self) -> CompletionResponse:
        if self.model not in OpenAIModel:
            return self._handle_error(
                exception=f"Model not supported. Please use one of the following models: {', '.join(OpenAIModel.list_values())}",
                error_type="ValueError",
            )
        try:
            messages = [{"role": "user", "content": self.prompt}]
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=self.temp,
            )
        except error.OpenAIError as e:
            logger.error(e.args[0])
            return self._handle_error(exception=e.args[0], error_type=type(e).__name__)
        except Exception as e:
            logger.error(e.args[0])
            # This might need to be changed to a different error
            raise Exception("Unknown OpenAI Error")

        return CompletionResponse(
            payload=response.choices[0].message["content"],
            message="OK",
            err="",
        )

    def _handle_error(self, exception: str, error_type: str) -> CompletionResponse:
        return CompletionResponse(message=exception, err=error_type)
