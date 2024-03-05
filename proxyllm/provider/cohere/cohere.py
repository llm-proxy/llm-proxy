from typing import Any, Dict

import cohere

from proxyllm.provider.base import BaseAdapter
from proxyllm.utils import logger, tokenizer
from proxyllm.utils.enums import BaseEnum
from proxyllm.utils.exceptions.provider import CohereException, UnsupportedModel

# Dictionary mapping Cohere model categories to task performance ratings.
cohere_category_data = {
    "model-categories": {
        "command": {
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
        "command-light": {
            "Code Generation Task": 3,
            "Text Generation Task": 2,
            "Translation and Multilingual Applications Task": 3,
            "Natural Language Processing Task": 2,
            "Conversational AI Task": 2,
            "Educational Applications Task": 3,
            "Healthcare and Medical Task": 4,
            "Legal Task": 4,
            "Financial Task": 4,
            "Content Recommendation Task": 3,
        },
        "command-nightly": {
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
        "command-light-nightly": {
            "Code Generation Task": 3,
            "Text Generation Task": 2,
            "Translation and Multilingual Applications Task": 3,
            "Natural Language Processing Task": 2,
            "Conversational AI Task": 2,
            "Educational Applications Task": 3,
            "Healthcare and Medical Task": 4,
            "Legal Task": 4,
            "Financial Task": 4,
            "Content Recommendation Task": 3,
        },
    }
}


class CohereAdapter(BaseAdapter):
    """
    Adapter class for the Cohere language model API.

    This adapter facilitates communication between the LLM Proxy application and the Cohere API,
    managing API requests and responses, error handling, and cost estimation based on token usage.

    Attributes:
        prompt (str): Default prompt to use for requests.
        model (str): Model identifier for the Cohere API.
        temperature (float): Temperature setting for text generation (creativity).
        api_key (str): API key for authenticating with the Cohere service.
        max_output_tokens (int | None): Maximum number of tokens for the model response.
        timeout (int): Timeout for API requests in seconds.
    """

    def __init__(
        self,
        prompt: str = "",
        model: str = "",
        temperature: float = 0.0,
        api_key: str = "",
        max_output_tokens: int | None = None,
        timeout: int = 120,  # default based on Cohere's API
    ) -> None:
        self.prompt = prompt
        self.model = model
        self.temperature = temperature
        self.api_key = api_key
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout

    def get_completion(self, prompt: str = "") -> str | None:
        """
        Requests a text completion from the Cohere model.

        Args:
            prompt (str): Input text prompt for the model.

        Returns:
            str | None: The text completion result from the model, or None if an error occurs.

        Raises:
            CohereException: If an error occurs during the API request.
        """
        try:
            co = cohere.Client(api_key=self.api_key, timeout=self.timeout)
            response = co.chat(
                max_tokens=self.max_output_tokens,
                message=prompt or self.prompt,
                model=self.model,
                temperature=self.temperature,
            )
            return response.text
        except cohere.CohereError as e:
            raise CohereException(exception=str(e), error_type="CohereError") from e
        except Exception as e:
            raise CohereException(
                exception=str(e), error_type="Unknown Cohere Error"
            ) from e

    def get_estimated_max_cost(
        self, prompt: str = "", price_data: Dict[str, Any] = None
    ) -> float:
        """
        Estimates the maximum cost for a given prompt based on token pricing.

        Args:
            prompt (str): Text prompt for which to estimate cost.
            price_data (Dict[str, Any]): Pricing data for tokens.

        Returns:
            float: Estimated maximum cost for the prompt.

        Raises:
            ValueError: If no prompt is provided.
        """
        if not self.prompt and not prompt:
            raise ValueError("No prompt provided.")

        # Assumption, model exists (check should be done at yml load level)
        logger.log(msg=f"MODEL: {self.model}", color="PURPLE")

        prompt_cost_per_token = price_data["prompt"]
        logger.log(msg=f"PROMPT (COST/TOKEN): {prompt_cost_per_token}")

        completion_cost_per_token = price_data["completion"]

        logger.log(msg=f"COMPLETION (COST/TOKEN): {completion_cost_per_token}")

        # Note: Avoiding costs for now
        # tokens = self.co.tokenize(text=prompt or self.prompt).tokens
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
        Retrieves the performance rank of the current model for a specified task category.

        Args:
            category (str): The task category to retrieve the rank for.

        Returns:
            int: Rank of the model in the specified category.
        """
        logger.log(msg=f"MODEL: {self.model}", color="PURPLE")
        logger.log(msg=f"CATEGORY OF PROMPT: {category}")

        category_rank = cohere_category_data["model-categories"][self.model][category]

        logger.log(msg=f"RANK OF PROMPT: {category_rank}", color="BLUE")

        return category_rank
