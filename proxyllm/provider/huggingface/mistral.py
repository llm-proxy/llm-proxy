import warnings

import requests
from tokenizers import Encoding

from proxyllm.provider.base import BaseAdapter, TokenizeResponse
from proxyllm.utils import logger, tokenizer
from proxyllm.utils.exceptions.provider import MistralException

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
        "mixtral-8x7b-instruct-v0.1": {
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
        self.generated_responses = []
        self.past_user_inputs = []
        self.response = None
        warnings.warn(
            """The class "mistral.py" under the provider "huggingface" is deprecated. Use "mistral.py" 
            under the "mistral" provider instead.""",
            DeprecationWarning,
        )

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
        warnings.warn(
            """The method named "get_completion()" in the "mistral.py" class found in the "huggingface" provider is deprecated. 
            Switch to using "mistral.py" under the "mistral" provider in order to use get_completion().""",
            DeprecationWarning,
        )

        if not self.api_key:
            raise ValueError("No Hugging Face API Key Provided")

        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            parameters = {
                "temperature": self.temperature,
                "max_length": self.max_output_tokens,
                "max_time": self.timeout,
            }

            def query(payload):
                response = requests.post(api_url, headers=headers, json=payload)
                return response.json()

            # For non-conversational mistral models
            if self.model != "mixtral-8x7b-instruct-v0.1":
                api_url = f"https://api-inference.huggingface.co/models/mistralai/{self.model}"
                output = query(
                    {
                        "inputs": prompt or self.prompt,
                        "parameters": parameters,
                    }
                )
                self.response = output[0]["generated_text"]
            # For conversational mistral models
            else:
                api_url = f"https://api-inference.huggingface.co/pipeline/conversational/mistralai/{self.model}"
                output = query(
                    {
                        "inputs": {
                            "past_user_inputs": self.past_user_inputs,
                            "generated_responses": self.generated_responses,
                            "text": prompt or self.prompt,
                        },
                        "parameters": parameters,
                    }
                )
                print(output)
                self.past_user_inputs = output["conversation"]["past_user_inputs"]
                self.generated_responses = output["conversation"]["generated_responses"]
                self.response = output["generated_text"]

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
        return self.response

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
        warnings.warn(
            """The method named "tokenize()" in the "mistral.py" class found in the "huggingface" provider is deprecated. 
            Switch to using "mistral.py" under the "mistral" provider in order to use tokenize().""",
            DeprecationWarning,
        )

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
        warnings.warn(
            """The method named "get_category_rank()" in the "mistral.py" class found in the "huggingface" provider is deprecated. 
            Switch to using "mistral.py" under the "mistral" provider in order to use get_category_rank().""",
            DeprecationWarning,
        )

        logger.log(msg=f"MODEL: {self.model}", color="PURPLE")
        logger.log(msg=f"CATEGORY OF PROMPT: {category}")

        category_rank = mistral_category_data["model-categories"][self.model][category]

        logger.log(msg=f"RANK OF PROMPT: {category_rank}", color="BLUE")

        return category_rank
