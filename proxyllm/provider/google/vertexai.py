from typing import Any, Dict, List, Tuple

from proxyllm.provider.base import BaseAdapter, TokenizeResponse
from proxyllm.utils import proxy_logger, timeout_function, tokenizer
from proxyllm.utils.enums import BaseEnum
from proxyllm.utils.exceptions.provider import VertexAIException

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
        "code-bison": {
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
        "codechat-bison": {
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
        "code-gecko": {
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


GEMINI_ROLE_MAPPINGS = {"user": "user", "assistant": "model"}
PALM_CODEY_ROLE_MAPPINGS = {"user": "user", "assistant": "bot"}


class ModelType(str, BaseEnum):
    GEMINI = ["gemini-pro"]
    CODEY = ["code-bison,codechat-bison,code-gecko"]
    PALM = ["text-bison", "chat-bison"]


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

    def _make_request(
        self,
        prompt: str,
        result: Dict[str, Any],
        chat_history: List[Dict[str, str]] | None = None,
    ):
        """
        Private method to make a request to the Vertex AI API.

        Args:
            prompt (str): The text prompt to send to the model.
            chat_history (List[Dict[str, str]]): The chat history for conversation

            result (Dict[str, Any]): Dictionary to store the output or exception.
        """
        if chat_history is None:
            chat_history = []

        try:
            from google.cloud import aiplatform

            aiplatform.init(project=self.project_id, location=self.location)
            parameters = {
                "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens,
            }

            provider_response = None

            # TODO :: Refactor so that there is one instance of the textgen/generative, language and chat models
            # will be changed in the future when more models are released

            # Handle GEMINI models
            if self.model in ModelType.GEMINI.value:
                from vertexai.preview.generative_models import GenerativeModel

                context, vertexai_chat_history = self.format_chat_history(
                    chat_history=chat_history, model_type="gemini"
                )

                chat_model = GenerativeModel(self.model)
                chat = chat_model.start_chat(history=vertexai_chat_history)
                response = chat.send_message(prompt or self.prompt)

                chat_history.append({"role": "user", "content": prompt or self.prompt})
                chat_history.append({"role": "assistant", "content": response.text})

                provider_response = {
                    "response": response.text,
                    "chat_history": chat_history,
                }

            # Handle Palm Models
            elif self.model in ModelType.PALM.value:
                if self.model == "text-bison":
                    from vertexai.language_models import TextGenerationModel

                    chat_model = TextGenerationModel.from_pretrained(self.model)
                    response = chat_model.predict(prompt or self.prompt, **parameters)
                    chat_history.append(
                        {"role": "user", "content": prompt or self.prompt}
                    )
                    chat_history.append({"role": "author", "content": response.text})
                    provider_response = {
                        "response": response.text,
                        "chat_history": chat_history,
                    }

                else:
                    from vertexai.language_models import ChatModel

                    context, vertexai_chat_history = self.format_chat_history(
                        chat_history=chat_history, model_type="palm"
                    )

                    chat_model = ChatModel.from_pretrained(self.model)
                    chat = chat_model.start_chat(
                        message_history=vertexai_chat_history, context=context
                    )
                    response = chat.send_message(prompt or self.prompt, **parameters)

                    chat_history.append(
                        {"role": "user", "content": prompt or self.prompt}
                    )
                    chat_history.append({"role": "assistant", "content": response.text})

                    provider_response = {
                        "response": response.text,
                        "chat_history": chat_history,
                    }

            # Handle Codey Models
            elif self.model in ModelType.CODEY.value:
                if self.model == "codechat-bison":
                    from vertexai.language_models import CodeChatModel

                    context, vertexai_chat_history = self.format_chat_history(
                        chat_history=chat_history, model_type="codey"
                    )
                    chat_model = CodeChatModel.from_pretrained(self.model)
                    chat = chat_model.start_chat(
                        message_history=vertexai_chat_history, context=context
                    )
                    response = chat.send_message(prompt or self.prompt, **parameters)

                    chat_history.append(
                        {"role": "user", "content": prompt or self.prompt}
                    )
                    chat_history.append({"role": "assistant", "content": response.text})
                    provider_response = {
                        "response": response.text,
                        "chat_history": chat_history,
                    }
                else:
                    from vertexai.language_models import CodeGenerationModel

                    chat_model = CodeGenerationModel.from_pretrained(self.model)
                    response = chat_model.predict(
                        prefix=prompt or self.prompt, **parameters
                    )
                    chat_history.append(
                        {"role": "user", "content": prompt or self.prompt}
                    )
                    chat_history.append({"role": "author", "content": response.text})
                    provider_response = {
                        "response": response.text,
                        "chat_history": chat_history,
                    }

            result["output"] = provider_response

        except Exception as e:
            result["exception"] = e

    def get_completion(
        self, prompt: str = "", chat_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any] | None:
        """
        Fetches text completion from the Vertex AI model.

        Args:
            prompt (str): The text prompt for generating completion.
            chat_history (List[Dict[str, str]]): The chat history for conversation

        Returns:
            Dict[str, Any] | None: The model's text response and chat history, or None if an error occurs.

        Raises:
            VertexAIException: If an error occurs during the API request.
        """
        result = {"output": None, "exception": None}

        if not self.force_timeout:
            self._make_request(
                prompt=prompt,
                result=result,
                chat_history=chat_history,
            )
        else:
            print(self.timeout)
            timeout_function.timeout_wrapper(
                self._make_request,
                self.timeout,
                prompt=prompt,
                result=result,
                chat_history=chat_history,
            )

        # We handle exception here so that it is picked up by logger
        if result["exception"]:
            raise VertexAIException(
                exception=result["exception"].args[0],
                error_type=type(result["exception"]).__name__,
            ) from result.get("exception", None)

        return result.get("output") or None

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
        proxy_logger.log(msg=f"MODEL: {self.model}", color="PURPLE")

        category_rank = vertexai_category_data["model-categories"][self.model][category]

        proxy_logger.log(msg=f"MODEL CATEGORY RANK: {category_rank}", color="BLUE")
        return category_rank

    def format_chat_history(
        self,
        chat_history: List[Dict[str, str]] | None = None,
        model_type: str = "",
    ) -> Tuple[str, List[Dict[str, str]]]:
        """
        Formats the chat history by separating system messages from user messages.
        The remaining messages are converted into ChatMessage objects based on the provided model type.

        Args:
            chat_history (List[Dict[str, str]], optional): A list of dictionaries representing the chat history.
                Each dictionary should contain 'role' and 'content' keys indicating the role of the speaker
                (system, user, or assistant) and the content of the message, respectively. Defaults to None.
            model_type (str, optional): The type of the model used for formatting the chat history.
                This determines the role mappings for the ChatMessage objects. Defaults to "".

        Returns:
            Tuple[str, List[Dict[str, str]]]: A tuple containing the system message (if present) and the formatted chat history.
                The system message is a string representing the context, and the formatted chat history is a list of dictionaries
                containing 'content' and 'author' keys, similar to the input but converted into ChatMessage objects.

        Notes:
            - System messages are extracted and separated from the chat history.
            - The original chat history is not modified; a deep copy is created for processing.
        """
        context = ""
        vertexai_chat_history = []

        if chat_history and chat_history[0].get("role") == "system":
            context = chat_history[0].get("content")
            chat_history = chat_history[1:]

        if model_type == "gemini":
            from vertexai.preview.generative_models import Content, Part

            for chat in chat_history:
                message = Content(
                    role=GEMINI_ROLE_MAPPINGS.get(chat["role"]),
                    parts=[Part.from_text(chat["content"])],
                )
                vertexai_chat_history.append(message)
        else:
            from vertexai.language_models import ChatMessage

            for chat in chat_history:
                message = ChatMessage(
                    content=chat["content"],
                    author=PALM_CODEY_ROLE_MAPPINGS.get(chat["role"]),
                )
                vertexai_chat_history.append(message)

        return context, vertexai_chat_history
