import copy
from typing import Any, Dict, List

from proxyllm.provider.base import BaseAdapter, TokenizeResponse
from proxyllm.utils import proxy_logger, tokenizer
from proxyllm.utils.exceptions.provider import MistralException

# Mapping of Mistral model categories to their task performance ratings.
mistral_category_data = {
    "model-categories": {
        "open-mistral-7b": {
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
        "open-mixtral-8x7b": {
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
        "mistral-small-latest": {
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
        "mistral-medium-latest": {
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
        "mistral-large-latest": {
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
    },
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

    def get_completion(
        self, prompt: str = "", chat_history: List[Dict[str, str]] | None = None
    ) -> Dict[str, Any] | None:
        """
        Requests a text completion from the specified Mistral model.

        Args:
            prompt (str): The text prompt for generating completion.
            chat_history (List[Dict[str, str]]): The chat history for conversation

        Returns:
            Dict[str, Any] | None: The model's text response and chat history, or None if an error occurs.

        Raises:
            MistralAPIStatusException: Returned when Mistral receives a non-200 response from the API.
            MistralAPIException: Returned when the API responds with an error message.
            MistralConnectionException: Returned when the SDK can not reach the API server for any reason
        """
        if not self.api_key:
            raise ValueError("No Mistral API Key Provided")

        if chat_history is None:
            chat_history = []

        from mistralai.client import MistralClient
        from mistralai.exceptions import (
            MistralAPIException,
            MistralAPIStatusException,
            MistralConnectionException,
        )

        try:
            client = MistralClient(api_key=self.api_key, timeout=self.timeout)

            mistral_chat_history = copy.deepcopy(chat_history)
            mistral_chat_history.append(
                {"role": "user", "content": prompt or self.prompt}
            )

            output = client.chat(
                max_tokens=self.max_output_tokens,
                messages=mistral_chat_history,
                model=self.model,
                temperature=self.temperature,
            )
            response_text = output.choices[0].message.content

            chat_history.append({"role": "user", "content": prompt or self.prompt})
            chat_history.append({"role": "assistant", "content": response_text})

            provider_response = {
                "response": response_text,
                "chat_history": chat_history,
            }

        except (
            MistralConnectionException,
            MistralAPIException,
            MistralAPIStatusException,
        ) as e:
            raise MistralException(exception=str(e), error_type="MistralError") from e
        except Exception as e:
            raise MistralException(
                exception=str(e), error_type="Unknown Mistral Error"
            ) from e

        return provider_response or None

    def tokenize(self, prompt: str = "") -> TokenizeResponse:
        """
        Tokenizes the provided prompt using the tokenizer.

        Args:
            prompt (str, optional): The prompt to be tokenized. Defaults to an empty string.

        Returns:
            TokenizeResponse: An object containing information about the tokenization process,
                including the number of input tokens and the maximum number of output tokens.

        Note:
            This method currently avoids calculating costs for tokenization.
        """
        encoding: Encoding = tokenizer.bpe_tokenize_encode(prompt or self.prompt)

        return TokenizeResponse(
            num_of_input_tokens=len(encoding.tokens),
            num_of_output_tokens=self.max_output_tokens or 256,
        )

    def get_category_rank(self, category: str = "") -> int:
        """
        Retrieves the performance rank of the current model for a specified category.

        Args:
            category (str): Task category to retrieve the model's rank.

        Returns:
            int: Performance rank of the model in the specified category.
        """
        proxy_logger.log(msg=f"MODEL: {self.model}", color="PURPLE")

        category_rank = mistral_category_data["model-categories"][self.model][category]

        proxy_logger.log(msg=f"MODEL CATEGORY RANK: {category_rank}", color="BLUE")

        return category_rank
