import requests

from llmproxy.provider.base import BaseAdapter
from llmproxy.utils import tokenizer
from llmproxy.utils.enums import BaseEnum
from llmproxy.utils.exceptions.provider import MistralException, UnsupportedModel
from llmproxy.utils.log import CustomLogger, logger

mistral_price_data = {
    "max-output-tokens": 50,
    "model-costs": {
        # Cost per 1k tokens * 1000
        "Mistral-7B-v0.1": {
            "prompt": 0.05 / 1_000_000,
            "completion": 0.25 / 1_000_000,
        },
        "Mistral-7B-Instruct-v0.2": {
            "prompt": 0.05 / 1_000_000,
            "completion": 0.25 / 1_000_000,
        },
        "Mistral-8x7B-Instruct-v0.1": {
            "prompt": 0.30 / 1_000_000,
            "completion": 1.0 / 1_000_000,
        },
    },
}

mistral_category_data = {
    "model-categories": {
        "Mistral-7B-v0.1": {
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
        "Mistral-8x7B-Instruct-v0.1": {
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
        "Mistral-7B-Instruct-v0.2": {
            "Code Generation Task": 2,
            "Text Generation Task": 1,
            "Translation and Multilingual Applications Task": 2,
            "Natural Language Processing Task": 2,
            "Conversational AI Task": 2,
            "Educational Applications Task": 1,
            "Healthcare and Medical Task": 1,
            "Legal Task": 4,
            "Financial Task": 4,
            "Content Recommendation Task": 3,
        },
    }
}


class MistralModel(str, BaseEnum):
    Mistral_7B_V01 = "Mistral-7B-v0.1"
    Mistral_7B_Instruct_V02 = "Mistral-7B-Instruct-v0.2"
    Mistral_8x7B_Instruct_V01 = "Mistral-8x7B-Instruct-v0.1"


class MistralAdapter(BaseAdapter):
    def __init__(
        self,
        prompt: str = "",
        model: MistralModel = MistralModel.Mistral_7B_V01.value,
        api_key: str | None = "",
        temperature: float = 1.0,
        max_output_tokens: int | None = None,
        timeout: int | None = None,
    ) -> None:
        self.prompt = prompt
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout

    def get_completion(self, prompt: str = "") -> str:
        if self.model not in MistralModel:
            raise UnsupportedModel(
                exception=f"Model not supported, please use one of the following: {', '.join(MistralModel.list_values())}",
                error_type="UnsupportedModel",
            )

        if not self.api_key:
            raise ValueError("No Hugging Face API Key Provided")

        try:
            api_url = (
                f"https://api-inference.huggingface.co/models/mistralai/{self.model}"
            )
            headers = {"Authorization": f"Bearer {self.api_key}"}

            def query(payload):
                response = requests.post(api_url, headers=headers, json=payload)
                return response.json()

            output = query(
                {
                    "inputs": prompt or self.prompt,
                    "parameters": {
                        "temperature": self.temperature,
                        "max_length": self.max_output_tokens,
                        "max_time": self.timeout,
                    },
                }
            )

        except requests.RequestException as e:
            raise MistralException(
                f"Request error: {e}", error_type="RequestError"
            ) from e
        except Exception as e:
            raise MistralException(
                f"Unknown error: {e}", error_type=" Unknown Mistral Error"
            ) from e

        # Output will be a dict if there is an error
        if "error" in output:
            raise MistralException(f"{output['error']}", error_type="MistralError")

        # Output will be a List[dict] if there is no error
        return output[0]["generated_text"]

    def get_estimated_max_cost(self, prompt: str = "") -> float:
        if not self.prompt and not prompt:
            logger.info("No prompt provided.")
            raise ValueError("No prompt provided.")

        # Assumption, model exists (check should be done at yml load level)
        logger.info(f"MODEL: {self.model}")

        prompt_cost_per_token = mistral_price_data["model-costs"][self.model]["prompt"]
        logger.info(f"PROMPT (COST/TOKEN): {prompt_cost_per_token}")

        completion_cost_per_token = mistral_price_data["model-costs"][self.model][
            "completion"
        ]
        logger.info(f"COMPLETION (COST/TOKEN): {completion_cost_per_token}")

        tokens = tokenizer.bpe_tokenize_encode(prompt or self.prompt)

        logger.info(f"INPUT TOKENS: {len(tokens)}")

        logger.info(f"COMPLETION TOKENS: {mistral_price_data['max-output-tokens']}")

        cost = round(
            prompt_cost_per_token * len(tokens)
            + completion_cost_per_token * mistral_price_data["max-output-tokens"],
            8,
        )

        logger.info(f"COST: {cost}")

        return cost

    def get_category_rank(self, category: str = "") -> int:
        logger.info(msg=f"Current model: {self.model}")
        logger.info(msg=f"Category of prompt: {category}")
        category_rank = mistral_category_data["model-categories"][self.model][category]
        logger.info(msg=f"Rank of category: {category_rank}")
        return category_rank
