from google.cloud import aiplatform
from vertexai.language_models import TextGenerationModel

from llmproxy.provider.base import BaseAdapter
from llmproxy.utils import timeout, tokenizer
from llmproxy.utils.enums import BaseEnum
from llmproxy.utils.exceptions.provider import UnsupportedModel, VertexAIException
from llmproxy.utils.log import CustomLogger, console_logger, file_logger

# VERTEX IS PER CHARACTER
vertexai_price_data = {
    "max-output-tokens": 50,
    "model-costs": {
        "prompt": 0.0005 / 1_000,
        "completion": 0.0005 / 1_000,
    },
}

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
    }
}


class VertexAIModel(str, BaseEnum):
    # Add other models
    PALM_TEXT = "text-bison"
    PALM_CHAT = "chat-bison"


class VertexAIAdapter(BaseAdapter):
    def __init__(
        self,
        prompt: str = "",
        temperature: float = 0,
        model: VertexAIModel = VertexAIModel.PALM_TEXT.value,
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
        try:
            aiplatform.init(project=self.project_id, location=self.location)
            parameters = {
                "prompt": prompt or self.prompt,
                "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens,
            }

            chat_model = TextGenerationModel.from_pretrained(self.model)

            response = chat_model.predict(**parameters)
            result["output"] = response.text

        except Exception as e:
            result["exception"] = e

    def get_completion(self, prompt: str = "") -> str | None:
        if self.model not in VertexAIModel:
            raise UnsupportedModel(
                exception=f"Model not supported. Please use one of the following models: {', '.join(VertexAIModel.list_values())}",
                error_type="ValueError",
            )

        result = {"output": None, "exception": None}

        if not self.force_timeout:
            self._make_request(prompt, result)
        else:
            timeout.timeout_wrapper(
                self._make_request, self.timeout, prompt=prompt, result=result
            )

        # We handle exception here so that it is picked up by logger
        if result["exception"]:
            raise VertexAIException(
                exception=result["exception"].args[0],
                error_type=type(result["exception"]).__name__,
            ) from result.get("exception", None)

        return result.get("output")

    def get_estimated_max_cost(self, prompt: str = "") -> float:
        if not self.prompt and not prompt:
            raise ValueError("No prompt provided.")

        # Assumption, model exists (check should be done at yml load level)

        file_logger.info(f"MODEL: {self.model}")
        console_logger.info(
            CustomLogger.CustomFormatter.purple
            + f"MODEL: {self.model}"
            + CustomLogger.CustomFormatter.reset
        )

        prompt_cost_per_character = vertexai_price_data["model-costs"]["prompt"]
        file_logger.info(f"PROMPT (COST/TOKEN): {prompt_cost_per_character}")
        console_logger.info(f"PROMPT (COST/TOKEN): {prompt_cost_per_character}")

        completion_cost_per_character = vertexai_price_data["model-costs"]["completion"]
        file_logger.info(f"COMPLETION (COST/TOKEN): {completion_cost_per_character}")
        console_logger.info(f"COMPLETION (COST/TOKEN): {completion_cost_per_character}")

        tokens = tokenizer.vertexai_encode(prompt or self.prompt)

        file_logger.info(f"INPUT TOKENS: {len(tokens)}")
        console_logger.info(f"INPUT TOKENS: {len(tokens)}")

        file_logger.info(
            f"COMPLETION TOKENS: {vertexai_price_data['max-output-tokens']}"
        )
        console_logger.info(
            f"COMPLETION TOKENS: {vertexai_price_data['max-output-tokens']}"
        )

        cost = round(
            prompt_cost_per_character * len(tokens)
            + completion_cost_per_character * vertexai_price_data["max-output-tokens"],
            8,
        )

        file_logger.info(f"COST: {cost}")
        console_logger.info(
            CustomLogger.CustomFormatter.green
            + f"COST: {cost}"
            + CustomLogger.CustomFormatter.reset
        )

        return cost

    def get_category_rank(self, category: str = "") -> int:
        file_logger.info(f"MODEL: {self.model}")
        console_logger.info(
            CustomLogger.CustomFormatter.purple
            + f"MODEL: {self.model}"
            + CustomLogger.CustomFormatter.reset
        )
        file_logger.info(f"CATEGORY OF PROMPT: {category}")
        console_logger.info(f"CATEGORY OF PROMPT: {category}")
        category_rank = vertexai_category_data["model-categories"][self.model][category]
        file_logger.info(f"RANK OF PROMPT: {category_rank}")
        console_logger.info(
            CustomLogger.CustomFormatter.blue
            + f"RANK OF PROMPT: {category_rank}"
            + CustomLogger.CustomFormatter.reset
        )
        return category_rank
