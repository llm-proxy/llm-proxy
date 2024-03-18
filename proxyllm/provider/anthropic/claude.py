from proxyllm.provider.base import BaseAdapter, TokenizeResponse

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
        model: str = "",
        max_output_tokens: int | None = None,
        timeout: int | None = None,
    ) -> None:
        self.prompt = prompt
        self.api_key = api_key
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout
