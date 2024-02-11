import cohere

from llmproxy.provider.base import BaseAdapter
from llmproxy.utils import tokenizer
from llmproxy.utils.enums import BaseEnum
from llmproxy.utils.exceptions.provider import CohereException, UnsupportedModel
from llmproxy.utils.log import CustomLogger, file_logger, console_logger

cohere_price_data_summarize_generate_chat = {
    "max-output-tokens": 50,
    "model-costs": {
        "prompt": 1.50 / 1_000_000,
        "completion": 2.00 / 1_000_000,
    },
}

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
    }
}


# These are the "Models" only for chat/command
class CohereModel(str, BaseEnum):
    COMMAND = "command"
    COMMAND_LIGHT = "command-light"
    COMMAND_NIGHTLY = "command-nightly"
    COMMAND_LIGHT_NIGHTLY = "command-light-nightly"


class CohereAdapter(BaseAdapter):
    def __init__(
        self,
        prompt: str = "",
        model: CohereModel = CohereModel.COMMAND.value,
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

    def get_completion(self, prompt: str = "") -> str | None:
        if self.model not in CohereModel:
            raise UnsupportedModel(
                exception=f"Model not supported. Please use one of the following models: {', '.join(CohereModel.list_values())}",
                error_type="UnsupportedModel",
            )

        try:
            co = cohere.Client(api_key=self.api_key, timeout=self.timeout)
            response = co.chat(
                max_tokens=self.max_output_tokens,
                message=prompt or self.prompt,
                model=self.model,
                temperature=self.temperature,
            )
            return response.text
        except cohere.CohereError as e:
            raise CohereException(exception=str(e), error_type="CohereError") from e
        except Exception as e:
            raise CohereException(
                exception=str(e), error_type="Unknown Cohere Error"
            ) from e

    def get_estimated_max_cost(self, prompt: str = "") -> float:
        if not self.prompt and not prompt:
            file_logger.info("No prompt provided.")
            console_logger.info("No prompt provided.")
            raise ValueError("No prompt provided.")

        # Assumption, model exists (check should be done at yml load level)

        file_logger.info(f"MODEL: {self.model}")
        console_logger.info(CustomLogger.CustomFormatter.purple + f"MODEL: {self.model}" + CustomLogger.CustomFormatter.reset)

        prompt_cost_per_token = cohere_price_data_summarize_generate_chat[
            "model-costs"
        ]["prompt"]
        file_logger.info(f"PROMPT (COST/TOKEN): {prompt_cost_per_token}")
        console_logger.info(f"PROMPT (COST/TOKEN): {prompt_cost_per_token}")

        completion_cost_per_token = cohere_price_data_summarize_generate_chat[
            "model-costs"
        ]["completion"]
        file_logger.info(f"COMPLETION (COST/TOKEN): {completion_cost_per_token}")
        console_logger.info(f"COMPLETION (COST/TOKEN): {completion_cost_per_token}")

        # Note: Avoiding costs for now
        # tokens = self.co.tokenize(text=prompt or self.prompt).tokens
        tokens = tokenizer.bpe_tokenize_encode(prompt or self.prompt)

        file_logger.info(f"INPUT TOKENS: {len(tokens)}")
        console_logger.info(f"INPUT TOKENS: {len(tokens)}")

        file_logger.info(f"COMPLETION TOKENS: {cohere_price_data_summarize_generate_chat['max-output-tokens']}")
        console_logger.info(f"COMPLETION TOKENS: {cohere_price_data_summarize_generate_chat['max-output-tokens']}")

        cost = round(
            prompt_cost_per_token * len(tokens)
            + completion_cost_per_token
            * cohere_price_data_summarize_generate_chat["max-output-tokens"],
            8,
        )

        file_logger.info(f"COST: {cost}")
        console_logger.info(CustomLogger.CustomFormatter.green + f"COST: {cost}" + CustomLogger.CustomFormatter.reset)

        return cost

    def get_category_rank(self, category: str = "") -> int:
        file_logger.info(msg=f"Current model: {self.model}")
        console_logger.info(msg=f"Current model: {self.model}")
        file_logger.info(msg=f"Category of prompt: {category}")
        console_logger.info(msg=f"Category of prompt: {category}")
        category_rank = cohere_category_data["model-categories"][self.model][category]
        file_logger.info(msg=f"Rank of category: {category_rank}")
        console_logger.info(msg=f"Rank of category: {category_rank}")
        return category_rank
