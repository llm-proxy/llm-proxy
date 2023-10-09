from models.base import BaseChatbot, CompletionResponse
from utils.enums import BaseEnum
import openai


class OpenAIModel(str, BaseEnum):
    GPT_4 = "gpt-4"
    GPT_4_32K = "gpt-4-32k"
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_3_5_TURBO_16K = "gpt-3.5-turbo-16k"


class OpenAI(BaseChatbot):
    def __init__(
        self,
        prompt: str = "",
        model: OpenAIModel = OpenAIModel.GPT_3_5_TURBO,
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
                exception="Model not supported", error_type="ValueError"
            )
        try:
            messages = [{"role": "user", "content": self.prompt}]
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=self.temp,
            )
        except openai.error.OpenAIError as e:
            return self._handle_error(exception=e, error_type=type(e).__name__)
        except Exception as e:
            # This might need to be changed to a different error
            raise Exception("Unknown Error")

        return CompletionResponse(
            payload=response.choices[0].message["content"],
            message="OK",
            err="",
        )

    def _handle_error(self, exception: str, error_type: str) -> CompletionResponse:
        return CompletionResponse(message=exception, err=error_type)
