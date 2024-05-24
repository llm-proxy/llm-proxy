from typing import Any, Dict, List

from tokenizers import Encoding

from proxyllm.provider.base import BaseAdapter, TokenizeResponse
from proxyllm.utils import proxy_logger, tokenizer
from proxyllm.utils.exceptions.provider import CohereException

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
        "command-r": {
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
        "command-r-plus": {
            "Code Generation Task": 3,
            "Text Generation Task": 3,
            "Translation and Multilingual Applications Task": 3,
            "Natural Language Processing Task": 4,
            "Conversational AI Task": 3,
            "Educational Applications Task": 3,
            "Healthcare and Medical Task": 3,
            "Legal Task": 3,
            "Financial Task": 3,
            "Content Recommendation Task": 3,
        },
    }
}

# Mapping from role names to Cohere's format
ROLE_NAME_TO_REP = {
    "user": "USER",
    "assistant": "CHATBOT",
    "system": "SYSTEM",
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

    def get_completion(
        self, prompt: str = "", chat_history: List[Dict[str, str]] | None = None
    ) -> Dict[str, Any] | None:
        """
        Requests a text completion from the Cohere model.

        Args:
            prompt (str): The text prompt for generating completion.
            chat_history (List[Dict[str, str]]): The chat history for conversation

        Returns:
            Dict[str, Any] | None: The model's text response and chat history, or None if an error occurs.

        Raises:
            CohereException: If an error occurs during the API request.
        """
        from cohere import Client

        try:
            if chat_history is None:
                chat_history = []

            # Convert the proxy chat history into a format that Cohere can process
            cohere_chat_history = self.format_chat_history(chat_history)

            co = Client(api_key=self.api_key, timeout=self.timeout)
            response = co.chat(
                max_tokens=self.max_output_tokens,
                message=prompt or self.prompt,
                model=self.model,
                temperature=self.temperature,
                chat_history=cohere_chat_history,
            )

            # append and return original to avoid reformatting
            chat_history.append({"role": "user", "content": self.prompt or prompt})
            chat_history.append({"role": "assistant", "content": response.text})

            provider_response = {
                "response": response.text,
                "chat_history": chat_history,
            }

        except Exception as e:
            raise CohereException(
                exception=str(e), error_type="Unknown Cohere Error"
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
        # Note: Avoiding costs for now
        # tokens = self.co.tokenize(text=prompt or self.prompt).tokens
        encoding: Encoding = tokenizer.bpe_tokenize_encode(prompt or self.prompt)

        return TokenizeResponse(
            num_of_input_tokens=len(encoding.tokens),
            num_of_output_tokens=self.max_output_tokens or 256,
        )

    def get_category_rank(self, category: str = "") -> int:
        """
        Retrieves the performance rank of the current model for a specified task category.

        Args:
            category (str): The task category to retrieve the rank for.

        Returns:
            int: Rank of the model in the specified category.
        """
        proxy_logger.log(msg=f"MODEL: {self.model}", color="PURPLE")

        category_rank = cohere_category_data["model-categories"][self.model][category]

        proxy_logger.log(msg=f"MODEL CATEGORY RANK: {category_rank}", color="BLUE")

        return category_rank

    def format_chat_history(self, chat_history):
        cohere_chat_history = []
        for chat in chat_history:
            cohere_chat_obj = {
                "role": ROLE_NAME_TO_REP.get(chat["role"]),
                "message": chat["content"],
            }
            cohere_chat_history.append(cohere_chat_obj)
        return cohere_chat_history
