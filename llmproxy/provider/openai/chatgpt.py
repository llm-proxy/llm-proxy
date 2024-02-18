import openai
import tiktoken

from llmproxy.provider.base import BaseAdapter
from llmproxy.utils import logger
from llmproxy.utils.enums import BaseEnum
from llmproxy.utils.exceptions.provider import OpenAIException, UnsupportedModel

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


class OpenAIAdapter(BaseAdapter):
    def __init__(
        self,
        prompt: str = "",
        model: OpenAIModel = OpenAIModel.GPT_3_5_TURBO_1106.value,
        temperature: float = 0,
        api_key: str = "",
        max_output_tokens: int | None = None,
        timeout: int | None = None,
    ) -> None:
        self.prompt = prompt
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.api_key = api_key
        self.timeout = timeout

    def get_completion(self, prompt: str = "") -> str | None:
        if self.model not in OpenAIModel:
            raise UnsupportedModel(
                exception=f"Model not supported. Please use one of the following models: {', '.join(OpenAIModel.list_values())}",
                error_type="UnsupportedModel",
            )

        # Prevent API Connection Error with empty API KEY
        if self.api_key == "":
            raise OpenAIException(
                exception="EMPTY API KEY: API key not provided",
                error_type="No API Key Provided",
            )

        try:
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt or self.prompt}],
                model=self.model,
                max_tokens=self.max_output_tokens,
                temperature=self.temperature,
                timeout=self.timeout,
            )

        except openai.OpenAIError as e:
            raise OpenAIException(
                exception=e.args[0], error_type=type(e).__name__
            ) from e
        except Exception as e:
            raise OpenAIException(
                exception=e.args[0], error_type="Unknown OpenAI Error"
            ) from e

        return response.choices[0].message.content or None

    def get_estimated_max_cost(self, prompt: str = "") -> float:
        if not self.prompt and not prompt:
            raise ValueError("No prompt provided.")

        # Assumption, model exists (check should be done at yml load level)
        encoder = tiktoken.encoding_for_model(self.model)

        logger.log(msg=f"MODEL: {self.model}", color="PURPLE")

        prompt_cost_per_token = open_ai_price_data["model-costs"][self.model]["prompt"]
        logger.log(msg=f"PROMPT (COST/TOKEN): {prompt_cost_per_token}")

        completion_cost_per_token = open_ai_price_data["model-costs"][self.model][
            "completion"
        ]
        logger.log(msg=f"COMPLETION (COST/TOKEN): {completion_cost_per_token}")
        tokens = encoder.encode(prompt or self.prompt)
        logger.log(msg=f"INPUT TOKENS: {len(tokens)}")
        logger.log(msg=f"COMPLETION TOKENS: {open_ai_price_data['max-output-tokens']}")

        cost = round(
            prompt_cost_per_token * len(tokens)
            + completion_cost_per_token * open_ai_price_data["max-output-tokens"],
            8,
        )
        logger.log(msg=f"COST: {cost}", color="GREEN")

        return cost

    def get_category_rank(self, category: str = "") -> int:
        logger.log(msg=f"MODEL: {self.model}", color="PURPLE")
        logger.log(msg=f"CATEGORY OF PROMPT: {category}")

        category_rank = open_ai_category_data["model-categories"][self.model][category]

        logger.log(msg=f"RANK OF PROMPT: {category_rank}", color="BLUE")

        return category_rank
