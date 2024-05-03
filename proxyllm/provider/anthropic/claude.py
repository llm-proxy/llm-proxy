from typing import Any, Dict, List, Tuple

from proxyllm.provider.base import BaseAdapter, TokenizeResponse
from proxyllm.utils import proxy_logger
from proxyllm.utils.exceptions.provider import AnthropicException

# TODO: Catagorization ratings for each model.
anthropic_category_data = {
    "model-categories": {
        "claude-3-opus-20240229": {
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
        "claude-3-sonnet-20240229": {
            "Code Generation Task": 1,
            "Text Generation Task": 1,
            "Translation and Multilingual Applications Task": 1,
            "Natural Language Processing Task": 1,
            "Conversational AI Task": 2,
            "Educational Applications Task": 1,
            "Healthcare and Medical Task": 2,
            "Legal Task": 2,
            "Financial Task": 2,
            "Content Recommendation Task": 1,
        },
        "claude-3-haiku-20240307": {
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


class ClaudeAdapter(BaseAdapter):
    """
    Adapter class for interacting with Anthropic's language models.

    Facilitates the sending of requests to and handling of responses from Anthropic's API,
    including authentication, setting request parameters, and parsing responses. This adapter
    is part of the ProxyLLM application, enabling seamless integration with Anthropic's services.

    Attributes:
        prompt (str): Default text prompt for generating responses.
        api_key (str): API key for authenticating requests to Anthropic models.
        auth_token (str): Authorization token for additional security, if required.
        temperature (float): Controls the randomness in the generated text, affecting creativity.
        model (str): Identifier for the selected Anthropic model.
        max_output_tokens (int): Maximum number of tokens for the response.
        timeout (int): Timeout for the API request in seconds.
    """

    def __init__(
        self,
        prompt: str = "",
        api_key: str | None = "",
        temperature: float = 0.0,
        model: str = "",
        max_output_tokens: int | None = None,
        timeout: int | None = None,
    ) -> None:
        self.prompt = prompt
        self.api_key = api_key
        self.temperature = temperature
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout

    def get_completion(
        self, prompt: str = "", chat_history: List[Dict[str, str]] | None = None
    ) -> Dict[str, Any] | None:
        """
        Requests a text completion from the specified Anthropic model.

        Args:
            prompt (str): The text prompt for generating completion.
            chat_history (List[Dict[str, str]]): The chat history for conversation

        Returns:
            Dict[str, Any] | None: The model's text response and chat history, or None if an error occurs.

        Raises:
            AnthropicException: If an API or internal error occurs during request processing.
        """
        if self.api_key == "":
            raise AnthropicException(
                exception="EMPTY API KEY: API key not provided",
                error_type="No API Key Provided",
            )

        if chat_history is None:
            chat_history = []

        try:
            from anthropic import Anthropic, AnthropicError

            # TODO :: Remove reinitialization of the client
            client = Anthropic(api_key=self.api_key)

            system_message, claude_chat_history = self.format_chat_history(
                chat_history=chat_history
            )
            claude_chat_history.append(
                {"role": "user", "content": prompt or self.prompt}
            )

            response = client.messages.create(
                messages=claude_chat_history,
                model=self.model,
                max_tokens=self.max_output_tokens,
                temperature=self.temperature,
                timeout=self.timeout,
                system=system_message,
            )
            response_text = response.content[0].text

            chat_history.append({"role": "user", "content": prompt or self.prompt})
            chat_history.append({"role": "assistant", "content": response_text})
            provider_response = {
                "response": response_text,
                "chat_history": chat_history,
            }

        except AnthropicError as e:
            raise AnthropicException(
                exception=e.args[0], error_type=type(e).__name__
            ) from e
        except Exception as e:
            raise AnthropicException(
                exception=e.args[0], error_type="Unknown Anthropic Error"
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
        Note: that this is only accurate for older models, e.g. `claude-2.1`. For newer
        models this can only be used as a _very_ rough estimate, instead you should rely
        on the `usage` property in the response for exact counts.
        """
        from anthropic import Anthropic

        client = Anthropic(api_key=self.api_key)

        return TokenizeResponse(
            num_of_input_tokens=client.count_tokens(prompt),
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

        category_rank = anthropic_category_data["model-categories"][self.model][
            category
        ]

        proxy_logger.log(msg=f"MODEL CATEGORY RANK: {category_rank}", color="BLUE")
        return category_rank

    def format_chat_history(
        self, chat_history: List[Dict[str, str]] = None
    ) -> Tuple[str, List[Dict[str, str]]]:
        """
        Formats the chat history by for claude by ignoring the system role and extracting it as a system message variable

        Args:
            chat_history (List[Dict[str, str]], optional): A list of dictionaries representing the chat history.
                Each dictionary contains 'role' and 'content' keys indicating the role of the speaker (system, user, or assistant)
                and the content of the message, respectively. Defaults to None.

        Returns:
            Tuple[str, List[Dict[str, str]]]: A tuple containing the system message (if present) and the formatted chat history.
                The system message is a string, and the formatted chat history is a list of dictionaries similar to the input.
        """
        import copy

        system_message = ""
        claude_chat_history = []

        if chat_history and chat_history[0].get("role") == "system":
            system_message = chat_history[0].get("content")
            claude_chat_history = copy.deepcopy(chat_history[1:])

        elif chat_history:
            claude_chat_history = copy.deepcopy(chat_history)

        return system_message, claude_chat_history
