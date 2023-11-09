from llmproxy.models.base import BaseModel, CompletionResponse
from llmproxy.utils.enums import BaseEnum
from llmproxy.utils.log import logger

import openai
from openai import error


class OpenAIModel(str, BaseEnum):
    GPT_4 = "gpt-4"
    GPT_4_32K = "gpt-4-32k"
    GPT_3_5_TURBO_1106 = "gpt-3.5-turbo-1106"
    GPT_3_5_TURBO_INSTRUCT = "gpt-3.5-turbo-instruct"


class OpenAI(BaseModel):
    def __init__(
        self,
        prompt: str = "",
        model: OpenAIModel = OpenAIModel.GPT_3_5_TURBO_1106.value,
        temperature: float = 0,
        api_key: str = "",
        max_output_tokens: int = None,
    ) -> None:
        self.prompt = prompt
        self.model = model
        self.temperature = temperature
        # We may have to pull this directly from .env and use different .env file/names for testing
        openai.api_key = api_key
        self.max_output_tokens = max_output_tokens

    def get_completion(self, prompt: str = "") -> CompletionResponse:
        if self.model not in OpenAIModel:
            return self._handle_error(
                exception=f"Model not supported. Please use one of the following models: {', '.join(OpenAIModel.list_values())}",
                error_type="ValueError",
            )
        try:
            messages = [
                {"role": "user", "content": prompt if prompt else self.prompt}
            ]
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
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
