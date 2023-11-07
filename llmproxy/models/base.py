from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import tiktoken
from llmproxy.utils.log import logger


@dataclass
class CompletionResponse:
    """
    payload: Data on successful response else ""
    message: Error message on unsuccessful response else "OK"
    err: Error type on unsuccessful response else ""
    """

    payload: str = ""
    message: str = ""
    err: str = ""


class BaseModel(ABC):
    """Abstract based class used to interface with (Language) Models.
    Current only one: Language.
    Likely more to be introduced if more types of LLMs are introduced (Speech, CNNs, Video...)
    """

    @abstractmethod
    def get_completion(self, **kwargs: Any) -> CompletionResponse:
        pass

    def get_estimated_max_cost(self) -> float:
        encoder = tiktoken.encoding_for_model("cl100k_base")

        if self.model not in OpenAIModel:
            print("hello")
            encoder = tiktoken.encoding_for_model("cl100k_base")

        logger.info(f"Tokenizing model: {self.model}")

        prompt_cost_per_token = openai_info["model-costs"][self.model]["prompt"] / (
            1000 * 10000
        )
        logger.info(f"Prompt Cost per token: {prompt_cost_per_token}")

        completion_cost_per_token = openai_info["model-costs"][self.model][
            "completion"
        ] / (1000 * 10000)
        logger.info(f"Output cost per token: {completion_cost_per_token}")

        tokens = encoder.encode(self.prompt)
        print(tokens)
        logger.info(f"Number of input tokens found: {len(tokens)}")

        logger.info(
            f"Final calculation using {len(tokens)} input tokens and {openai_info['max-output-tokens']} output tokens"
        )

        cost = round(
            prompt_cost_per_token * len(tokens)
            + completion_cost_per_token * openai_info["max-output-tokens"],
            8,
        )

        logger.info(f"Calculated Cost: {cost}")

        return cost
