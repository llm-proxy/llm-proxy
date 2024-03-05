from typing import Any, Dict

import openai
import tiktoken

from proxyllm.provider.base import BaseAdapter
from proxyllm.utils import logger
from proxyllm.utils.enums import BaseEnum
from proxyllm.utils.exceptions.provider import OpenAIException, UnsupportedModel

# Mapping of OpenAI model categories to their respective task performance ratings.
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


class OpenAIAdapter(BaseAdapter):
    """
    Adapter class for interacting with OpenAI's language models.

    Facilitates the sending of requests to and the handling of responses from OpenAI's API,
    including authentication, setting request parameters, and parsing responses. This adapter
    is part of the LLM Proxy application, enabling seamless integration with OpenAI's services.

    Attributes:
        prompt (str): Default text prompt for generating responses.
        model (str): Identifier for the selected OpenAI model.
        temperature (float): Temperature parameter controlling the creativity of the response.
        api_key (str): API key for authenticating requests to OpenAI.
        max_output_tokens (int): Maximum number of tokens for the response.
        timeout (int): Timeout for the API request in seconds.
    """

    def __init__(
        self,
        prompt: str = "",
        model: str = "",
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
        """
        Requests a text completion from the specified OpenAI model.

        Args:
            prompt (str): The text prompt for generating completion.

        Returns:
            str | None: The model's text response, or None if an error occurs.

        Raises:
            OpenAIException: If an API or internal error occurs during request processing.
        """

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

    def get_estimated_max_cost(
        self, prompt: str = "", price_data: Dict[str, Any] = None
    ) -> float:
        """
        Estimates the maximum cost for processing a given prompt based on token pricing.

        Args:
            prompt (str): The text prompt for which to estimate the cost.
            price_data (Dict[str, Any]): Pricing data for tokens.

        Returns:
            float: The estimated cost for processing the prompt.

        Raises:
            ValueError: If no prompt is provided and no default prompt is set.
        """
        if not self.prompt and not prompt:
            raise ValueError("No prompt provided.")

        # Assumption, model exists (check should be done at yml load level)
        encoder = tiktoken.encoding_for_model(self.model)

        logger.log(msg=f"MODEL: {self.model}", color="PURPLE")

        prompt_cost_per_token = price_data["prompt"]
        logger.log(msg=f"PROMPT (COST/TOKEN): {prompt_cost_per_token}")

        completion_cost_per_token = price_data["completion"]
        logger.log(msg=f"COMPLETION (COST/TOKEN): {completion_cost_per_token}")

        tokens = encoder.encode(prompt or self.prompt)
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
            category (str): The task category for which to retrieve the model's rank.

        Returns:
            int: The performance rank of the model in the specified category.
        """
        logger.log(msg=f"MODEL: {self.model}", color="PURPLE")
        logger.log(msg=f"CATEGORY OF PROMPT: {category}")

        category_rank = open_ai_category_data["model-categories"][self.model][category]

        logger.log(msg=f"RANK OF PROMPT: {category_rank}", color="BLUE")

        return category_rank
