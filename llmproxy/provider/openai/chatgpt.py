from llmproxy.provider.base import BaseModel, CompletionResponse
from llmproxy.utils.enums import BaseEnum
from llmproxy.utils.log import logger
import openai
from openai import error
import tiktoken


# This should be available later from the yaml file
# Cost is converted into whole numbers to avoid inconsistent floats
open_ai_price_data = {
    "max-output-tokens": 50,
    "model-costs": {
        # Cost per 1k tokens * 1000
        "gpt-3.5-turbo-1106": {
            "prompt": 0.0010 / 1000,
            "completion": 0.0020 / 1000,
        },
        "gpt-3.5-turbo-instruct": {
            "prompt": 0.0015 / 1000,
            "completion": 0.0020 / 1000,
        },
        "gpt-4": {
            "prompt": 0.03 / 1000,
            "completion": 0.06 / 1000,
        },
        "gpt-4-32k": {
            "prompt": 0.06 / 1000,
            "completion": 0.12 / 1000,
        },
        "gpt-4-1106-preview": {
            "prompt": 0.01 / 1000,
            "completion": 0.03 / 1000,
        },
        "gpt-4-1106-vision-preview": {
            "prompt": 0.01 / 1000,
            "completion": 0.03 / 1000,
        },
    },
}


class OpenAIModel(str, BaseEnum):
    GPT_4_1106_PREVIEW = "gpt-4-1106-preview"
    GPT_4_1106_VISION_PREVIEW = "gpt-4-1106-vision-preview"
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
            messages = [{"role": "user", "content": prompt or self.prompt}]
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

    def get_estimated_max_cost(self, prompt: str = "") -> float:
        if not self.prompt and not prompt:
            logger.info("No prompt provided.")
            raise ValueError("No prompt provided.")

        # Assumption, model exists (check should be done at yml load level)
        encoder = tiktoken.encoding_for_model(self.model)

        logger.info(f"Tokenizing model: {self.model}")

        prompt_cost_per_token = open_ai_price_data["model-costs"][self.model]["prompt"]
        logger.info(f"Prompt Cost per token: {prompt_cost_per_token}")

        completion_cost_per_token = open_ai_price_data["model-costs"][self.model][
            "completion"
        ]
        logger.info(f"Output cost per token: {completion_cost_per_token}")

        tokens = encoder.encode(prompt or self.prompt)

        logger.info(f"Number of input tokens found: {len(tokens)}")

        logger.info(
            f"Final calculation using {len(tokens)} input tokens and {open_ai_price_data['max-output-tokens']} output tokens"
        )

        cost = round(
            prompt_cost_per_token * len(tokens)
            + completion_cost_per_token * open_ai_price_data["max-output-tokens"],
            8,
        )

        logger.info(f"Calculated Cost: {cost}")

        return cost

    def _handle_error(self, exception: str, error_type: str) -> CompletionResponse:
        return CompletionResponse(message=exception, err=error_type)
