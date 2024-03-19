from proxyllm.provider.base import BaseAdapter, TokenizeResponse
from proxyllm.utils.exceptions.provider import AnthropicException
from anthropic import Anthropic
claude_category_data = {
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
        temperature: float | None = None,
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
        if self.api_key == "":
            raise AnthropicException(
                exception="EMPTY API KEY: API key not provided",
                error_type="No API Key Provided",
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
                exception=e.args[0], error_type="Unknown OpenAI Error"
            ) from e

        return response.choices[0].message.content or None


