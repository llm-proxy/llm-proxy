from typing import Any, Dict, List

import requests
from tokenizers import Encoding

from proxyllm.provider.base import BaseAdapter, TokenizeResponse
from proxyllm.utils import proxy_logger, tokenizer
from proxyllm.utils.exceptions.provider import EmptyPrompt, Llama2Exception

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

    def get_completion(
        self, prompt: str = "", chat_history: List[Dict[str, str]] | None = None
    ) -> Dict[str, Any] | None:
        """
        Requests a text completion from the specified Llama-2 model.

        Args:
            prompt (str): Text prompt for generating completion.
            chat_history (List[Dict[str, str]]): The chat history for conversation

        Returns:
            Dict[str, Any] | None: The model's text response and chat history, or None if an error occurs.

        Raises:
            Llama2Exception: If an API or internal error occurs during request processing.
            EmptyPrompt: If no prompt is provided and the instance has no default prompt.
        """
        if not self.api_key:
            raise Llama2Exception(exception="No API Provided", error_type="ValueError")

        if self.prompt == "" and prompt == "":
            raise EmptyPrompt("Empty prompt detected")

        if chat_history is None:
            chat_history = []

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
            output_text = output[0]["generated_text"]

            chat_history.append({"role": "user", "content": self.prompt or prompt})
            chat_history.append({"role": "assistant", "content": output_text})

            provider_response = {
                "response": output_text,
                "chat_history": chat_history,
            }

        except Exception as e:
            raise Llama2Exception(exception=e.args[0], error_type="Llama2Error") from e

        if output["error"]:
            raise Llama2Exception(exception=output["error"], error_type="Llama2Error")

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
            category (str): Task category for which to retrieve the model's rank.

        Returns:
            int: Performance rank of the model in the specified category.
        """
        proxy_logger.log(msg=f"MODEL: {self.model}", color="PURPLE")

        category_rank = llama2_category_data["model-categories"][self.model][category]

        proxy_logger.log(msg=f"MODEL CATEGORY RANK: {category_rank}", color="BLUE")

        return category_rank
