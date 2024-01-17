from google.cloud import aiplatform
from google.auth import exceptions as auth_exceptions
from google.api_core import exceptions as api_exceptions
from llmproxy.provider.base import BaseProvider, CompletionResponse
from llmproxy.utils.enums import BaseEnum
from llmproxy.utils.log import logger
from vertexai.language_models import TextGenerationModel
from llmproxy.utils import tokenizer

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

    def get_completion(self, prompt: str = "") -> CompletionResponse:
        if self.model not in VertexAIModel:
            return self._handle_error(
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
            logger.error(e.args[0])
            return self._handle_error(exception=e.args[0], error_type=type(e).__name__)
        except auth_exceptions.GoogleAuthError as e:
            logger.error(e.args[0])
            return self._handle_error(exception=e.args[0], error_type=type(e).__name__)
        except Exception as e:
            logger.error(e.args[0])
            raise Exception(f"Unknown Vertexai Error:{e}")

        return CompletionResponse(payload=output, message="OK", err="")

    def get_estimated_max_cost(self, prompt: str = "") -> float:
        if not self.prompt and not prompt:
            logger.info("No prompt provided.")
            raise ValueError("No prompt provided.")

        # Assumption, model exists (check should be done at yml load level)

        logger.info(f"Tokenizing model: {self.model}")

        prompt_cost_per_character = vertexai_price_data["model-costs"]["prompt"]
        logger.info(f"Prompt Cost per token: {prompt_cost_per_character}")

        completion_cost_per_character = vertexai_price_data["model-costs"]["completion"]
        logger.info(f"Output cost per token: {completion_cost_per_character}")

        tokens = tokenizer.vertexai_encode(prompt or self.prompt)

        logger.info(f"Number of input tokens found: {len(tokens)}")

        logger.info(
            f"Final calculation using {len(tokens)} input tokens and {vertexai_price_data['max-output-tokens']} output tokens"
        )

        cost = round(
            prompt_cost_per_character * len(tokens)
            + completion_cost_per_character * vertexai_price_data["max-output-tokens"],
            8,
        )

        logger.info(f"Calculated Cost: {cost}")

        return cost

    def get_category_rank(self, category: str = "") -> str:
        logger.info(msg=f"Current model: {self.model}")
        logger.info(msg=f"Category of prompt: {category}")
        category_rank = vertexai_category_data["model-categories"][self.model][category]
        logger.info(msg=f"Rank of category: {category_rank}")
        return category_rank

    def _handle_error(self, exception: str, error_type: str) -> CompletionResponse:
        return CompletionResponse(message=exception, err=error_type)
