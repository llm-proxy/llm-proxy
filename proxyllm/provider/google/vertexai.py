from proxyllm.provider.base import BaseAdapter, TokenizeResponse
from typing import Any, Dict

from proxyllm.provider.base import BaseAdapter
from proxyllm.utils import logger, timeout_function, tokenizer
from proxyllm.utils.exceptions.provider import VertexAIException

# VERTEX IS PER CHARACTER
vertexai_price_data = {
    "max-output-tokens": 50,
    "model-costs": {
        "text-bison": {
            "prompt": 0.0005 / 1_000,
            "completion": 0.0005 / 1_000,
        },
        "chat-bison": {
            "prompt": 0.0005 / 1_000,
            "completion": 0.0005 / 1_000,
        },
        "gemini-pro": {
            "prompt": 0.0000125 / 1_000,
            "completion": 0.000375 / 1_000,
        },
    },
}
>>>>>>> bc76c28 (implement gemini)
>>>>>>> 0eab6b9 (add text bison):llmproxy/provider/google/vertexai.py

# Dictionary mapping Vertex AI model categories to task performance ratings.
vertexai_category_data = {
    "model-categories": {
        "text-bison": {
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
        "chat-bison": {
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
        "gemini-pro": {
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


class VertexAIAdapter(BaseAdapter):
    """
    Adapter class for the Vertex AI language models API.

    Manages API requests, responses, error handling, and cost estimation for token usage in the context of the LLM Proxy application.

    Attributes:
        prompt (str): Default text prompt for model requests.
        temperature (float): Temperature for response generation, affecting creativity.
        model (str): Identifier for the selected Vertex AI model.
        project_id (str): Google Cloud project ID.
        location (str): Google Cloud location for the AI Platform.
        max_output_tokens (int): Maximum number of tokens for the response.
        timeout (int): Timeout for the API request.
        force_timeout (bool): Whether to enforce a timeout for the request.
    """

    def __init__(
        self,
        prompt: str = "",
        temperature: float = 0,
        model: str = "",
        project_id: str | None = "",
        location: str | None = "us-central1",
        max_output_tokens: int | None = None,
        timeout: int | None = None,
        force_timeout: bool = False,
    ) -> None:
        self.prompt = prompt
        self.temperature = temperature
        self.model = model
        self.project_id = project_id
        self.location = location
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout
        self.force_timeout = force_timeout

    def _make_request(self, prompt, result):
        """
        Private method to make a request to the Vertex AI API.

        Args:
            prompt (str): The text prompt to send to the model.
            result (Dict[str, Any]): Dictionary to store the output or exception.
        """
        try:
            from google.cloud import aiplatform

            aiplatform.init(project=self.project_id, location=self.location)
            parameters = {
                "prompt": prompt or self.prompt,
                "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens,
            }

            if self.model == "gemini-pro":
                from vertexai.preview.generative_models import GenerativeModel, Part

                chat_model = GenerativeModel(self.model)
                response = chat_model.generate_content(prompt or self.prompt)
            else:
                from vertexai.language_models import TextGenerationModel

                chat_model = TextGenerationModel.from_pretrained(self.model)
                response = chat_model.predict(**parameters)
            result["output"] = response.text

        except Exception as e:
            result["exception"] = e

    def get_completion(self, prompt: str = "") -> str | None:
        """
        Fetches text completion from the Vertex AI model.

        Args:
            prompt (str): Text prompt for generating completion.

        Returns:
            str | None: The model's text response, or None if an error occurred.

        Raises:
            VertexAIException: If an error occurs during the API request.
        """
        result = {"output": None, "exception": None}

        if not self.force_timeout:
            self._make_request(prompt, result)
        else:
            timeout_function.timeout_wrapper(
                self._make_request, self.timeout, prompt=prompt, result=result
            )

        # We handle exception here so that it is picked up by logger
        if result["exception"]:
            raise VertexAIException(
                exception=result["exception"].args[0],
                error_type=type(result["exception"]).__name__,
            ) from result.get("exception", None)

        return result.get("output")

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

        tokens = tokenizer.vertexai_encode(prompt or self.prompt)

        return TokenizeResponse(
            num_of_input_tokens=len(tokens),
            num_of_output_tokens=self.max_output_tokens or 256,
        )

    def get_category_rank(self, category: str = "") -> int:
        """
        Retrieves the model's performance rank for a specified task category.

        Args:
            category (str): The task category to check.

        Returns:
            int: Rank of the model in the specified category.
        """
        logger.log(msg=f"MODEL: {self.model}", color="PURPLE")
        logger.log(msg=f"CATEGORY OF PROMPT: {category}")

        category_rank = vertexai_category_data["model-categories"][self.model][category]

        logger.log(msg=f"RANK OF PROMPT: {category_rank}", color="BLUE")
        return category_rank
