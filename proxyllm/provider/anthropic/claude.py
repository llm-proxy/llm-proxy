from proxyllm.provider.base import BaseAdapter, TokenizeResponse
from proxyllm.utils import logger
from proxyllm.utils.exceptions.provider import AnthropicException

# TODO: Catagorization ratings for each model.
anthropic_category_data = {
    "model-categories": {
        "claude-3-opus-20240229": {
            "Code Generation Task": 1,
            "Text Generation Task": 1,
            "Translation and Multilingual Applications Task": 1,
            "Natural Language Processing Task": 1,
            "Conversational AI Task": 1,
            "Educational Applications Task": 1,
            "Healthcare and Medical Task": 1,
            "Legal Task": 1,
            "Financial Task": 1,
            "Content Recommendation Task": 1,
        },
        "claude-3-sonnet-20240229": {
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
        "claude-3-haiku-20240307": {
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


class ClaudeAdapter(BaseAdapter):
    def __init__(
        self,
        prompt: str = "",
        api_key: str | None = "",
        auth_token: str | None = "",
        temperature: float = 0,
        model: str = "",
        max_output_tokens: int | None = None,
        timeout: int | None = None,
    ) -> None:
        self.prompt = prompt
        self.api_key = api_key
        self.auth_token = auth_token
        self.temperature = temperature
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout

    def get_completion(self, prompt: str = "") -> str | None:
        if self.api_key == "" and self.auth_token == "":
            raise AnthropicException(
                exception="API key or auth token not provided",
                error_type="No API Key or Auth Token Provided",
            )

        from anthropic import Anthropic, AnthropicError

        try:
            client = Anthropic(api_key=self.api_key)
            response = client.messages.create(
                messages=[{"role": "user", "content": prompt or self.prompt}],
                model=self.model,
                max_tokens=self.max_output_tokens,
                temperature=self.temperature,
                timeout=self.timeout,
            )
        except AnthropicError as e:
            raise AnthropicException(
                exception=e.args[0], error_type=type(e).__name__
            ) from e
        except Exception as e:
            raise AnthropicException(
                exception=e.args[0], error_type="Unknown Anthropic Error"
            ) from e
        return response.content[0].text or None

    def tokenize(self, prompt: str = "") -> TokenizeResponse:
        pass

    def get_category_rank(self, category: str = "") -> int:
        logger.log(msg=f"MODEL: {self.model}", color="PURPLE")
        logger.log(msg=f"CATEGORY OF PROMPT: {category}")

        category_rank = anthropic_category_data["model-categories"][self.model][
            category
        ]

        logger.log(msg=f"RANK OF PROMPT: {category_rank}", color="BLUE")
        return category_rank
