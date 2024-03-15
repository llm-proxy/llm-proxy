from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict


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
    def get_completion(self, prompt: str = "") -> str | None:
        """
        Abstract method to retrieve completion for a given prompt.

        Parameters:
        - prompt (str): The prompt for which completion is requested. Defaults to an empty string.

        Returns:
        - str or None: The completion for the provided prompt, or None if no completion is available.
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
