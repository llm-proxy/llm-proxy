from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class TokenizeResponse:
    num_of_input_tokens: int
    num_of_output_tokens: int


class BaseAdapter(ABC):
    """Abstract base class used to interface with (Language) Models.
    Current only one: Language.
    Likely more to be introduced if more types of LLMs are introduced (Speech, CNNs, Video...)
    """

    @abstractmethod
    def get_completion(
        self, prompt: str = "", chat_history: List[Dict[str, str]] | None = None
    ) -> Dict[str, Any] | None:
        """
        Abstract method to retrieve completion for a given prompt.

        Parameters:
        - prompt (str): The prompt for which completion is requested. Defaults to an empty string.
        - chat_history (List[Dict[str, str]]): The chat history for conversation. Defaults to None.

        Returns:
        - Dict[str, Any] or None: The model's text response and chat history, or None if an error occurs.
        """

    @abstractmethod
    def tokenize(self, prompt="") -> TokenizeResponse:
        pass

    @abstractmethod
    def get_category_rank(self, category: str = "") -> int:
        """
        Abstract method to retrieve the rank of a given category.

        Parameters:
        - category (str): The category for which the rank is requested. Defaults to an empty string.

        Returns:
        - int: The rank of the specified category.
        """
