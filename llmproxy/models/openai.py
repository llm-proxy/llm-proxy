import tiktoken
from llmproxy.models.base import BaseChatbot, CompletionResponse
from llmproxy.utils.enums import BaseEnum
from llmproxy.utils.log import logger
import openai
from openai import error


# This should be available later from the yaml file
# Cost is converted into whole numbers to avoid inconsistent floats
openai_info = {
    "max-output-tokens": 256,
    "model-costs": {
        # Cost per 1000 * 10000 tokens
        "gpt-3.5-turbo-4k": {
            "prompt": 15,
            "completion": 20,
        },
        "gpt-3.5-turbo-16k": {
            "prompt": 30,
            "completion": 40,
        },
        "gpt-4": {
            "prompt": 300,
            "completion": 600,
        },
        "gpt-4-8k": {
            "prompt": 300,
            "completion": 600,
        },
        "gpt-4-32k": {
            "prompt": 600,
            "completion": 1200,
        },
        "text-embedding-ada-002-v2": {
            "prompt": 1,
            "completion": 1,
        },
    },
}


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

    def get_estimated_max_cost(self) -> float:
        encoder = tiktoken.encoding_for_model("cl100k_base")

        if self.model not in OpenAIModel:
            encoder = tiktoken.encoding_for_model("cl100k_base")

        logger.info(f"Tokenizing model: {self.model}")

        prompt_cost_per_token = openai_info["model-costs"][self.model]["prompt"] / (
            1000 * 10000
        )
        logger.info(f"Prompt Cost per token: {prompt_cost_per_token}")

        completion_cost_per_token = openai_info["model-costs"][self.model][
            "completion"
        ] / (1000 * 10000)
        logger.info(f"Output cost per token: {completion_cost_per_token}")

        tokens = encoder.encode(self.prompt)
        print(tokens)
        logger.info(f"Number of input tokens found: {len(tokens)}")

        logger.info(
            f"Final calculation using {len(tokens)} input tokens and {openai_info['max-output-tokens']} output tokens"
        )

        cost = round(
            prompt_cost_per_token * len(tokens)
            + completion_cost_per_token * openai_info["max-output-tokens"],
            8,
        )

        logger.info(f"Calculated Cost: {cost}")

        return cost

    def _handle_error(self, exception: str, error_type: str) -> CompletionResponse:
        return CompletionResponse(message=exception, err=error_type)
