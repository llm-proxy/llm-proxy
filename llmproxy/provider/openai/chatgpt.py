import openai
import tiktoken
from openai import error

from llmproxy.llmproxy import load_model_costs
from llmproxy.provider.base import BaseProvider
from llmproxy.utils.enums import BaseEnum
from llmproxy.utils.exceptions.provider import OpenAIException, UnsupportedModel
from llmproxy.utils.log import logger

# This should be available later from the yaml file
# Cost is converted into whole numbers to avoid inconsistent floats
open_ai_price_data = load_model_costs("llmproxy/config/internal.config.yml", "OpenAI")

open_ai_category_data = {
    "model-categories": {
        "gpt-3.5-turbo-1106": {
            "Code Generation Task": 2,
            "Text Generation Task": 1,
            "Translation and Multilingual Applications Task": 2,
            "Natural Language Processing Task": 1,
            "Conversational AI Task": 1,
            "Educational Applications Task": 2,
            "Healthcare and Medical Task": 3,
            "Legal Task": 3,
            "Financial Task": 3,
            "Content Recommendation Task": 2,
        },
        "gpt-3.5-turbo-instruct": {
            "Code Generation Task": 2,
            "Text Generation Task": 1,
            "Translation and Multilingual Applications Task": 2,
            "Natural Language Processing Task": 1,
            "Conversational AI Task": 1,
            "Educational Applications Task": 2,
            "Healthcare and Medical Task": 3,
            "Legal Task": 3,
            "Financial Task": 3,
            "Content Recommendation Task": 2,
        },
        "gpt-4": {
            "Code Generation Task": 1,
            "Text Generation Task": 1,
            "Translation and Multilingual Applications Task": 1,
            "Natural Language Processing Task": 1,
            "Conversational AI Task": 1,
            "Educational Applications Task": 1,
            "Healthcare and Medical Task": 2,
            "Legal Task": 2,
            "Financial Task": 2,
            "Content Recommendation Task": 1,
        },
        "gpt-4-32k": {
            "Code Generation Task": 1,
            "Text Generation Task": 1,
            "Translation and Multilingual Applications Task": 1,
            "Natural Language Processing Task": 1,
            "Conversational AI Task": 1,
            "Educational Applications Task": 1,
            "Healthcare and Medical Task": 2,
            "Legal Task": 2,
            "Financial Task": 2,
            "Content Recommendation Task": 1,
        },
    }
}


class OpenAIModel(str, BaseEnum):
    GPT_4_1106_PREVIEW = "gpt-4-1106-preview"
    GPT_4_1106_VISION_PREVIEW = "gpt-4-1106-vision-preview"
    GPT_4 = "gpt-4"
    GPT_4_32K = "gpt-4-32k"
    GPT_3_5_TURBO_1106 = "gpt-3.5-turbo-1106"
    GPT_3_5_TURBO_INSTRUCT = "gpt-3.5-turbo-instruct"


class OpenAI(BaseProvider):
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

    def get_completion(self, prompt: str = "") -> str:
        if self.model not in OpenAIModel:
            raise UnsupportedModel(
                exception=f"Model not supported. Please use one of the following models: {', '.join(OpenAIModel.list_values())}",
                error_type="OpenAI Error",
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
            raise OpenAIException(
                exception=e.args[0], error_type=type(e).__name__
            ) from e
        except Exception as e:
            raise OpenAIException(
                exception=e.args[0], error_type="Unknown OpenAI Error"
            ) from e

        return response.choices[0].message["content"]

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

    def get_category_rank(self, category: str = "") -> str:
        logger.info(msg=f"Current model: {self.model}")
        logger.info(msg=f"Category of prompt: {category}")
        category_rank = open_ai_category_data["model-categories"][self.model][category]
        logger.info(msg=f"Rank of category: {category_rank}")
        return category_rank
