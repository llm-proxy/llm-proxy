from typing import Any, Dict, List

import tiktoken

from proxyllm.provider.base import BaseAdapter, TokenizeResponse
from proxyllm.utils import proxy_logger
from proxyllm.utils.exceptions.provider import OpenAIException

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
        "gpt-3.5-turbo-0125": {
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
        "gpt-4-0125-preview": {
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
        "gpt-4-1106-preview": {
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
        "gpt-4-turbo": {
            "Code Generation Task": 3,
            "Text Generation Task": 4,
            "Translation and Multilingual Applications Task": 4,
            "Natural Language Processing Task": 5,
            "Conversational AI Task": 3,
            "Educational Applications Task": 3,
            "Healthcare and Medical Task": 3,
            "Legal Task": 3,
            "Financial Task": 4,
            "Content Recommendation Task": 5,
        },
        "gpt-4o": {
            "Code Generation Task": 4,
            "Text Generation Task": 4,
            "Translation and Multilingual Applications Task": 4,
            "Natural Language Processing Task": 5,
            "Conversational AI Task": 4,
            "Educational Applications Task": 3,
            "Healthcare and Medical Task": 3,
            "Legal Task": 3,
            "Financial Task": 4,
            "Content Recommendation Task": 5,
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
        self.api_key = api_key
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout

    def get_completion(
        self, prompt: str = "", chat_history: List[Dict[str, str]] | None = None
    ) -> Dict[str, Any] | None:
        """
        Requests a text completion from the specified OpenAI model.

        Args:
            prompt (str): The text prompt for generating completion.
            chat_history (List[Dict[str, str]]): The chat history for conversation

        Returns:
            Dict[str, Any] | None: The model's text response and chat history, or None if an error occurs.

        Raises:
            OpenAIException: If an API or internal error occurs during request processing.
        """

        # Prevent API Connection Error with empty API KEY
        if self.api_key == "":
            raise OpenAIException(
                exception="EMPTY API KEY: API key not provided",
                error_type="No API Key Provided",
            )

        if chat_history is None:
            chat_history = []

        import copy

        from openai import OpenAI, OpenAIError

        try:
            client = OpenAI(api_key=self.api_key)

            openai_chat_history = copy.deepcopy(chat_history)
            openai_chat_history.append(
                {"role": "user", "content": prompt or self.prompt}
            )
            response = client.chat.completions.create(
                messages=openai_chat_history,
                model=self.model,
                max_tokens=self.max_output_tokens,
                temperature=self.temperature,
                timeout=self.timeout,
            )
            response_text = response.choices[0].message.content

            chat_history.append({"role": "user", "content": prompt or self.prompt})
            chat_history.append({"role": "assistant", "content": response_text})

            provider_response = {
                "response": response_text,
                "chat_history": chat_history,
            }
        except OpenAIError as e:
            raise OpenAIException(
                exception=e.args[0], error_type=type(e).__name__
            ) from e
        except Exception as e:
            raise OpenAIException(
                exception=e.args[0], error_type="Unknown OpenAI Error"
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
        encoder = tiktoken.encoding_for_model(self.model)

        tokens = encoder.encode(prompt or self.prompt)

        return TokenizeResponse(
            num_of_input_tokens=len(tokens),
            num_of_output_tokens=self.max_output_tokens or 256,
        )

    def get_category_rank(self, category: str = "") -> int:
        """
        Retrieves the performance rank of the current model for a specified task category.

        Args:
            category (str): The task category for which to retrieve the model's rank.

        Returns:
            int: The performance rank of the model in the specified category.
        """
        proxy_logger.log(msg=f"MODEL: {self.model}", color="PURPLE")

        category_rank = open_ai_category_data["model-categories"][self.model][category]

        proxy_logger.log(msg=f"MODEL CATEGORY RANK: {category_rank}", color="BLUE")

        return category_rank
