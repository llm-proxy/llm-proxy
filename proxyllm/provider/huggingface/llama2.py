from typing import Any, Dict

import requests

from proxyllm.provider.base import BaseAdapter
from proxyllm.utils import logger, tokenizer
from proxyllm.utils.enums import BaseEnum
from proxyllm.utils.exceptions.provider import (
    EmptyPrompt,
    Llama2Exception,
    UnsupportedModel,
)

# Mapping of Llama-2 model categories to their respective task performance ratings.
llama2_category_data = {
    "model-categories": {
        "llama-2-7b-chat-hf": {
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
        "llama-2-13b-chat-hf": {
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
        "llama-2-70b-chat-hf": {
            "Code Generation Task": 1,
            "Text Generation Task": 1,
            "Translation and Multilingual Applications Task": 1,
            "Natural Language Processing Task": 1,
            "Conversational AI Task": 1,
            "Educational Applications Task": 1,
            "Healthcare and Medical Task": 1,
            "Legal Task": 1,
            "Financial Task": 1,
            "Content Recommendation Task": 1,
        },
        "llama-2-7b-chat": {
            "Code Generation Task": 4,
            "Text Generation Task": 5,
            "Translation and Multilingual Applications Task": 4,
            "Natural Language Processing Task": 5,
            "Conversational AI Task": 5,
            "Educational Applications Task": 4,
            "Healthcare and Medical Task": 3,
            "Legal Task": 3,
            "Financial Task": 3,
            "Content Recommendation Task": 4,
        },
        "llama-2-7b-hf": {
            "Code Generation Task": 4,
            "Text Generation Task": 5,
            "Translation and Multilingual Applications Task": 4,
            "Natural Language Processing Task": 5,
            "Conversational AI Task": 5,
            "Educational Applications Task": 4,
            "Healthcare and Medical Task": 3,
            "Legal Task": 3,
            "Financial Task": 3,
            "Content Recommendation Task": 4,
        },
        "llama-2-7b": {
            "Code Generation Task": 4,
            "Text Generation Task": 5,
            "Translation and Multilingual Applications Task": 4,
            "Natural Language Processing Task": 5,
            "Conversational AI Task": 5,
            "Educational Applications Task": 4,
            "Healthcare and Medical Task": 3,
            "Legal Task": 3,
            "Financial Task": 3,
            "Content Recommendation Task": 4,
        },
        "llama-2-13b-chat": {
            "Code Generation Task": 3,
            "Text Generation Task": 3,
            "Translation and Multilingual Applications Task": 3,
            "Natural Language Processing Task": 3,
            "Conversational AI Task": 3,
            "Educational Applications Task": 3,
            "Healthcare and Medical Task": 3,
            "Legal Task": 3,
            "Financial Task": 3,
            "Content Recommendation Task": 3,
        },
        "llama-2-13b-hf": {
            "Code Generation Task": 3,
            "Text Generation Task": 3,
            "Translation and Multilingual Applications Task": 3,
            "Natural Language Processing Task": 3,
            "Conversational AI Task": 3,
            "Educational Applications Task": 3,
            "Healthcare and Medical Task": 3,
            "Legal Task": 3,
            "Financial Task": 3,
            "Content Recommendation Task": 3,
        },
        "llama-2-13b": {
            "Code Generation Task": 3,
            "Text Generation Task": 3,
            "Translation and Multilingual Applications Task": 3,
            "Natural Language Processing Task": 3,
            "Conversational AI Task": 3,
            "Educational Applications Task": 3,
            "Healthcare and Medical Task": 3,
            "Legal Task": 3,
            "Financial Task": 3,
            "Content Recommendation Task": 3,
        },
        "llama-2-70b-chat": {
            "Code Generation Task": 3,
            "Text Generation Task": 3,
            "Translation and Multilingual Applications Task": 3,
            "Natural Language Processing Task": 3,
            "Conversational AI Task": 3,
            "Educational Applications Task": 3,
            "Healthcare and Medical Task": 3,
            "Legal Task": 3,
            "Financial Task": 3,
            "Content Recommendation Task": 3,
        },
        "llama-2-70b-hf": {
            "Code Generation Task": 3,
            "Text Generation Task": 3,
            "Translation and Multilingual Applications Task": 3,
            "Natural Language Processing Task": 3,
            "Conversational AI Task": 3,
            "Educational Applications Task": 3,
            "Healthcare and Medical Task": 3,
            "Legal Task": 3,
            "Financial Task": 3,
            "Content Recommendation Task": 3,
        },
        "llama-2-70b": {
            "Code Generation Task": 3,
            "Text Generation Task": 3,
            "Translation and Multilingual Applications Task": 3,
            "Natural Language Processing Task": 3,
            "Conversational AI Task": 3,
            "Educational Applications Task": 3,
            "Healthcare and Medical Task": 3,
            "Legal Task": 3,
            "Financial Task": 3,
            "Content Recommendation Task": 3,
        },
    }
}


class Llama2Adapter(BaseAdapter):
    """
    Adapter class for interacting with the Llama-2 API.

    This adapter encapsulates the logic for sending requests to and handling responses from Llama-2 language models.
    It supports setting system-level prompts, handling API keys, and managing request parameters like temperature
    and token limits.

    Attributes:
        system_prompt (str): System-level instruction to guide the model's responses.
        prompt (str): Default user-provided text prompt for generating responses.
        api_key (str): API key for authenticating requests to the Llama-2 API.
        temperature (float): Controls the randomness of the generated text.
        model (str): Identifier for the Llama-2 model being used.
        max_output_tokens (int): Maximum number of tokens in the generated text.
        timeout (int): Time limit in seconds for the API response.
    """

    def __init__(
        self,
        prompt: str = "",
        system_prompt: str = "Answer politely",
        api_key: str | None = "",
        temperature: float = 1.0,
        model: str = "",
        max_output_tokens: int | None = None,
        timeout: int | None = None,
    ) -> None:
        self.system_prompt = system_prompt
        self.prompt = prompt
        self.api_key = api_key
        self.temperature = temperature
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout

    def get_completion(self, prompt: str = "") -> str | None:
        """
        Requests a text completion from the specified Llama-2 model.

        Args:
            prompt (str): Text prompt for generating completion.

        Returns:
            str | None: The text completion from the model, or None if an error occurs.

        Raises:
            Llama2Exception: If an API or internal error occurs during request processing.
            EmptyPrompt: If no prompt is provided and the instance has no default prompt.
        """
        if not self.api_key:
            raise Llama2Exception(exception="No API Provided", error_type="ValueError")

        if self.prompt == "" and prompt == "":
            raise EmptyPrompt("Empty prompt detected")

        try:
            api_url = (
                f"https://api-inference.huggingface.co/models/meta-llama/{self.model}"
            )
            headers = {"Authorization": f"Bearer {self.api_key}"}

            def query(payload):
                response = requests.post(api_url, headers=headers, json=payload)
                return response.json()

            # Llama2 prompt template
            prompt_template = f"<s>[INST] <<SYS>>\n{{{{ {self.system_prompt} }}}}\n<</SYS>>\n{{{{ {prompt or self.prompt} }}}}\n[/INST]"

            output = query(
                {
                    "inputs": prompt_template,
                    "parameters": {
                        "max_length": self.max_output_tokens,
                        "temperature": self.temperature,
                        "max_time": self.timeout,
                    },
                }
            )

        except Exception as e:
            raise Llama2Exception(exception=e.args[0], error_type="Llama2Error") from e

        if output["error"]:
            raise Llama2Exception(exception=output["error"], error_type="Llama2Error")

        return output[0]["generated_text"]

    def get_estimated_max_cost(
        self, prompt: str = "", price_data: Dict[str, Any] = None
    ) -> float:
        """
        Estimates the cost for a given prompt based on predefined pricing data.

        Args:
            prompt (str): Text prompt for cost estimation.
            price_data (Dict[str, Any]): Cost per token for prompt and completion.

        Returns:
            float: Estimated cost for processing the given prompt.

        Raises:
            ValueError: If no prompt is provided and no default prompt is set.
        """
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

    def get_category_rank(self, category: str = "") -> int:
        """
        Retrieves the performance rank of the current model for a specified category.

        Args:
            category (str): Task category for which to retrieve the model's rank.

        Returns:
            int: Performance rank of the model in the specified category.
        """
        logger.log(msg=f"MODEL: {self.model}", color="PURPLE")
        logger.log(msg=f"CATEGORY OF PROMPT: {category}")

        category_rank = llama2_category_data["model-categories"][self.model][category]

        logger.log(msg=f"RANK OF PROMPT: {category_rank}", color="BLUE")

        return category_rank
