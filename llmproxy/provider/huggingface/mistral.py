from typing import Any, Dict

import requests
from huggingface_hub import InferenceClient
from llmproxy.provider.base import BaseAdapter
from llmproxy.utils import logger, tokenizer
from llmproxy.utils.enums import BaseEnum
from llmproxy.utils.exceptions.provider import MistralException, UnsupportedModel

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


class MistralAdapter(BaseAdapter):
    def __init__(
        self,
        prompt: str = "",
        model: str = "",
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
        self.generated_responses = []
        self.past_user_inputs = []

    def get_completion(self, prompt: str = "") -> str:
        if not self.api_key:
            raise ValueError("No Hugging Face API Key Provided")

        try:
            API_URL = "https://api-inference.huggingface.co/pipeline/conversational/facebook/blenderbot-400M-distill"
            headers = {"Authorization": f"Bearer {self.api_key}"}

            """For mistral use: mistralai/{self.model}"""
            # API_URL = "https://api-inference.huggingface.co/pipeline/conversational/mistralai/Mixtral-8x7B-Instruct-v0.1"

            def query(payload):
                response = requests.post(API_URL, headers=headers, json=payload)
                return response.json()

            output = query(
                {
                    "inputs": {
                        "past_user_inputs": self.past_user_inputs,
                        "generated_responses": self.generated_responses,
                        "text": prompt or self.prompt,
                    },
                }
            )
            print(output)
            self.past_user_inputs = output["conversation"]["past_user_inputs"]
            self.generated_responses = output["conversation"]["generated_responses"]

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
        return output["generated_text"]

    def get_estimated_max_cost(
        self, prompt: str = "", price_data: Dict[str, Any] = None
    ) -> float:
        if not self.prompt and not prompt:
            raise ValueError("No prompt provided.")

        # Assumption, model exists (check should be done at yml load level)
        logger.log(msg=f"MODEL: {self.model}", color="PURPLE")

        prompt_cost_per_token = price_data["prompt"]
        logger.log(msg=f"PROMPT (COST/TOKEN): {prompt_cost_per_token}")

        completion_cost_per_token = price_data["completion"]
        logger.log(msg=f"COMPLETION (COST/TOKEN): {completion_cost_per_token}")

        tokens = tokenizer.bpe_tokenize_encode(prompt or self.prompt)
        logger.log(msg=f"INPUT TOKENS: {len(tokens)}")
        logger.log(msg=f"COMPLETION TOKENS: {self.max_output_tokens}")

        cost = round(
            prompt_cost_per_token * len(tokens)
            + completion_cost_per_token * self.max_output_tokens,
            8,
        )

        logger.log(msg=f"COST: {cost}", color="GREEN")

        return cost

    def clear_chat(self) -> None:
        self.generated_responses = []
        self.past_user_inputs = []

    def get_category_rank(self, category: str = "") -> int:
        logger.log(msg=f"MODEL: {self.model}", color="PURPLE")
        logger.log(msg=f"CATEGORY OF PROMPT: {category}")

        category_rank = mistral_category_data["model-categories"][self.model][category]

        logger.log(msg=f"RANK OF PROMPT: {category_rank}", color="BLUE")

        return category_rank
