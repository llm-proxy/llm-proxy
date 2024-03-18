from proxyllm.provider.base import BaseAdapter, TokenizeResponse

claude_category_data = {
    "model-categories": {
        "claude-3-opus-20240229": {
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
  

