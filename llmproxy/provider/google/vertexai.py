from cohere.client import logger
from typing import Any, Dict

from google.cloud import aiplatform
from vertexai.language_models import TextGenerationModel

from llmproxy.provider.base import BaseAdapter
from llmproxy.utils import logger, timeout, tokenizer
from llmproxy.utils.enums import BaseEnum
from llmproxy.utils.exceptions.provider import UnsupportedModel, VertexAIException

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

    def get_estimated_max_cost(
        self, prompt: str = "", price_data: Dict[str, Any] = None
    ) -> float:
        if not self.prompt and not prompt:
            raise ValueError("No prompt provided.")

        # Assumption, model exists (check should be done at yml load level)
        logger.log(msg=f"MODEL: {self.model}", color="PURPLE")

        prompt_cost_per_character = price_data["prompt"]
        logger.log(msg=f"PROMPT (COST/CHARACTER): {prompt_cost_per_character}")

        completion_cost_per_character = price_data["completion"]
        logger.log(msg=f"COMPLETION (COST/CHARACTER): {completion_cost_per_character}")

        tokens = tokenizer.vertexai_encode(prompt or self.prompt)

        logger.log(msg=f"INPUT TOKENS: {len(tokens)}")

        logger.log(msg=f"COMPLETION TOKENS: {self.max_output_tokens}")

        cost = round(
            prompt_cost_per_character * len(tokens)
            + completion_cost_per_character * self.max_output_tokens,
            8,
        )

        logger.log(msg=f"COST: {cost}", color="GREEN")

        return cost

    def get_category_rank(self, category: str = "") -> int:
        logger.log(msg=f"MODEL: {self.model}", color="PURPLE")
        logger.log(msg=f"CATEGORY OF PROMPT: {category}")

        category_rank = vertexai_category_data["model-categories"][self.model][category]

        logger.log(msg=f"RANK OF PROMPT: {category_rank}", color="BLUE")
        return category_rank
