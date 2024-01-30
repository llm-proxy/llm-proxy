from google.api_core import exceptions as api_exceptions
from google.auth import exceptions as auth_exceptions
from google.cloud import aiplatform
from vertexai.language_models import TextGenerationModel

from llmproxy.provider.base import BaseProvider
from llmproxy.utils import tokenizer
from llmproxy.utils.enums import BaseEnum
from llmproxy.utils.exceptions.provider import UnsupportedModel, VertexAIException
from llmproxy.utils.log import logger

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


class VertexAI(BaseProvider):
    def __init__(
        self,
        prompt: str = "",
        temperature: float = 0,
        model: VertexAIModel = VertexAIModel.PALM_TEXT.value,
        project_id: str | None = "",
        location: str | None = "us-central1",
        max_output_tokens: int = None,
    ) -> None:
        self.prompt = prompt
        self.temperature = temperature
        self.model = model
        self.project_id = project_id
        self.location = location
        self.max_output_tokens = max_output_tokens

    def get_completion(self, prompt: str = "") -> str:
        if self.model not in VertexAIModel:
            raise UnsupportedModel(
                exception=f"Model not supported Please use one of the following models: {', '.join(VertexAIModel.list_values())}",
                error_type="ValueError",
            )
        try:
            aiplatform.init(project=self.project_id, location=self.location)
            # TODO developer - override these parameters as needed:
            parameters = {
                # Temperature controls the degree of randomness in token selection.
                "temperature": self.temperature,
                # Token limit determines the maximum amount of text output.
                "max_output_tokens": self.max_output_tokens,
            }

            chat_model = TextGenerationModel.from_pretrained(self.model)
            response = chat_model.predict(prompt or self.prompt, **parameters)
            output = response.text

        except api_exceptions.GoogleAPIError as e:
            raise VertexAIException(
                exception=e.args[0], error_type=type(e).__name__
            ) from e
        except auth_exceptions.GoogleAuthError as e:
            raise VertexAIException(
                exception=e.args[0], error_type=type(e).__name__
            ) from e
        except Exception as e:
            raise VertexAIException(
                exception=e.args[0], error_type=type(e).__name__
            ) from e

        return output

    def get_estimated_max_cost(self, prompt: str = "") -> float:
        if not self.prompt and not prompt:
            logger.info("No prompt provided.")
            raise ValueError("No prompt provided.")

        # Assumption, model exists (check should be done at yml load level)

        logger.info(f"MODEL: {self.model}")

        prompt_cost_per_character = vertexai_price_data["model-costs"]["prompt"]
        logger.info(f"PROMPT (COST/TOKEN): {prompt_cost_per_character}")

        completion_cost_per_character = vertexai_price_data["model-costs"]["completion"]
        logger.info(f"COMPLETION (COST/TOKEN): {completion_cost_per_character}")

        tokens = tokenizer.vertexai_encode(prompt or self.prompt)

        logger.info(f"INPUT TOKENS: {len(tokens)}")

        logger.info(
            f"COMPLETION TOKENS: {vertexai_price_data['max-output-tokens']}"
        )

        cost = round(
            prompt_cost_per_character * len(tokens)
            + completion_cost_per_character * vertexai_price_data["max-output-tokens"],
            8,
        )

        logger.info(f"COST: {cost}")

        return cost

    def get_category_rank(self, category: str = "") -> str:
        logger.info(msg=f"Current model: {self.model}")
        logger.info(msg=f"Category of prompt: {category}")
        category_rank = vertexai_category_data["model-categories"][self.model][category]
        logger.info(msg=f"Rank of category: {category_rank}")
        return category_rank
