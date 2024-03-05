from typing import Any, Dict

import requests

from proxyllm.provider.base import BaseAdapter
from proxyllm.utils import logger, tokenizer
from proxyllm.utils.enums import BaseEnum
from proxyllm.utils.exceptions.provider import MistralException, UnsupportedModel

# Mapping of Mistral model categories to their task performance ratings.
mistral_category_data = {
    "model-categories": {
        "mistral-7b-v0.1": {
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
        "mistral-8x7b-instruct-v0.1": {
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
        "mistral-7b-instruct-v0.2": {
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
    """
    Adapter class for the Mistral language models API.

    Encapsulates the logic for sending requests to and handling responses from Mistral language models,
    including API authentication, parameter management, and response parsing.

    Attributes:
        prompt (str): Default text prompt for generating responses.
        model (str): Identifier for the Mistral model being used.
        api_key (str): API key for authenticating requests to Mistral.
        temperature (float): Temperature for controlling the creativity of the response.
        max_output_tokens (int): Maximum number of tokens for the generated response.
        timeout (int): Timeout for the API request in seconds.
    """

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

    def get_completion(self, prompt: str = "") -> str:
        """
        Requests a text completion from the specified Mistral model.

        Args:
            prompt (str): Text prompt for generating completion.

        Returns:
            str: The text completion from the model.

        Raises:
            MistralException: If an API or internal error occurs during request processing.
            ValueError: If no API key is provided.
        """
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
            category (str): Task category to retrieve the model's rank.

        Returns:
            int: Performance rank of the model in the specified category.
        """
        logger.log(msg=f"MODEL: {self.model}", color="PURPLE")
        logger.log(msg=f"CATEGORY OF PROMPT: {category}")

        category_rank = mistral_category_data["model-categories"][self.model][category]

        logger.log(msg=f"RANK OF PROMPT: {category_rank}", color="BLUE")

        return category_rank
