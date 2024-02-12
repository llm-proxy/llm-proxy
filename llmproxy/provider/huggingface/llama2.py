import requests
from typing import Any, Dict
from llmproxy.provider.base import BaseAdapter
from llmproxy.utils import tokenizer
from llmproxy.utils.enums import BaseEnum
from llmproxy.utils.exceptions.provider import (
    EmptyPrompt,
    Llama2Exception,
    UnsupportedModel,
)
from llmproxy.utils.log import logger

llama2_category_data = {
    "model-categories": {
        "Llama-2-7b-chat-hf": {
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
        "Llama-2-13b-chat-hf": {
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
        "Llama-2-70b-chat-hf": {
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
        "Llama-2-7b-chat": {
            "Code Generation Task": 4,
            "Text Generation Task": 5,
            "Translation and Multilingual Applications Task": 4,
            "Natural Language Processing Task": 5,
            "Conversational AI Task": 5,
            "Educational Applications Task": 4,
            "Healthcare and Medical Task": 3,
            "Legal Task": 3,
            "Financial Task": 3,
            "Content Recommendation Task": 4,
        },
        "Llama-2-7b-hf": {
            "Code Generation Task": 4,
            "Text Generation Task": 5,
            "Translation and Multilingual Applications Task": 4,
            "Natural Language Processing Task": 5,
            "Conversational AI Task": 5,
            "Educational Applications Task": 4,
            "Healthcare and Medical Task": 3,
            "Legal Task": 3,
            "Financial Task": 3,
            "Content Recommendation Task": 4,
        },
        "Llama-2-7b": {
            "Code Generation Task": 4,
            "Text Generation Task": 5,
            "Translation and Multilingual Applications Task": 4,
            "Natural Language Processing Task": 5,
            "Conversational AI Task": 5,
            "Educational Applications Task": 4,
            "Healthcare and Medical Task": 3,
            "Legal Task": 3,
            "Financial Task": 3,
            "Content Recommendation Task": 4,
        },
        "Llama-2-13b-chat": {
            "Code Generation Task": 3,
            "Text Generation Task": 3,
            "Translation and Multilingual Applications Task": 3,
            "Natural Language Processing Task": 3,
            "Conversational AI Task": 3,
            "Educational Applications Task": 3,
            "Healthcare and Medical Task": 3,
            "Legal Task": 3,
            "Financial Task": 3,
            "Content Recommendation Task": 3,
        },
        "Llama-2-13b-hf": {
            "Code Generation Task": 3,
            "Text Generation Task": 3,
            "Translation and Multilingual Applications Task": 3,
            "Natural Language Processing Task": 3,
            "Conversational AI Task": 3,
            "Educational Applications Task": 3,
            "Healthcare and Medical Task": 3,
            "Legal Task": 3,
            "Financial Task": 3,
            "Content Recommendation Task": 3,
        },
        "Llama-2-13b": {
            "Code Generation Task": 3,
            "Text Generation Task": 3,
            "Translation and Multilingual Applications Task": 3,
            "Natural Language Processing Task": 3,
            "Conversational AI Task": 3,
            "Educational Applications Task": 3,
            "Healthcare and Medical Task": 3,
            "Legal Task": 3,
            "Financial Task": 3,
            "Content Recommendation Task": 3,
        },
        "Llama-2-70b-chat": {
            "Code Generation Task": 3,
            "Text Generation Task": 3,
            "Translation and Multilingual Applications Task": 3,
            "Natural Language Processing Task": 3,
            "Conversational AI Task": 3,
            "Educational Applications Task": 3,
            "Healthcare and Medical Task": 3,
            "Legal Task": 3,
            "Financial Task": 3,
            "Content Recommendation Task": 3,
        },
        "Llama-2-70b-hf": {
            "Code Generation Task": 3,
            "Text Generation Task": 3,
            "Translation and Multilingual Applications Task": 3,
            "Natural Language Processing Task": 3,
            "Conversational AI Task": 3,
            "Educational Applications Task": 3,
            "Healthcare and Medical Task": 3,
            "Legal Task": 3,
            "Financial Task": 3,
            "Content Recommendation Task": 3,
        },
        "Llama-2-70b": {
            "Code Generation Task": 3,
            "Text Generation Task": 3,
            "Translation and Multilingual Applications Task": 3,
            "Natural Language Processing Task": 3,
            "Conversational AI Task": 3,
            "Educational Applications Task": 3,
            "Healthcare and Medical Task": 3,
            "Legal Task": 3,
            "Financial Task": 3,
            "Content Recommendation Task": 3,
        },
    }
}


class Llama2Model(str, BaseEnum):
    LLAMA_2_7B_CHAT_HF = "Llama-2-7b-chat-hf"
    LLAMA_2_7B_CHAT = "Llama-2-7b-chat"
    LLAMA_2_7B_HF = "Llama-2-7b-hf"
    LLAMA_2_7B = "Llama-2-7b"
    LLAMA_2_13B_CHAT_HF = "Llama-2-13b-chat-hf"
    LLAMA_2_13B_CHAT = "Llama-2-13b-chat"
    LLAMA_2_13B_HF = "Llama-2-13b-hf"
    LLAMA_2_13B = "Llama-2-13b"
    LLAMA_2_70B_CHAT_HF = "Llama-2-70b-chat-hf"
    LLAMA_2_70B_CHAT = "Llama-2-70b-chat"
    LLAMA_2_70B_HF = "Llama-2-70b-hf"
    LLAMA_2_70B = "Llama-2-70b"


class Llama2Adapter(BaseAdapter):
    def __init__(
        self,
        prompt: str = "",
        system_prompt: str = "Answer politely",
        api_key: str | None = "",
        temperature: float = 1.0,
        model: Llama2Model = Llama2Model.LLAMA_2_7B_CHAT_HF.value,
        max_output_tokens: int | None = None,
        timeout: int | None = None,
    ) -> None:
        self.system_prompt = system_prompt
        self.prompt = prompt
        self.api_key = api_key
        self.temperature = temperature
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout

    def get_completion(self, prompt: str = "") -> str | None:
        if not self.api_key:
            raise Llama2Exception(exception="No API Provided", error_type="ValueError")

        if self.prompt == "" and prompt == "":
            raise EmptyPrompt("Empty prompt detected")

        if self.model not in Llama2Model:
            raise UnsupportedModel(
                exception=f"Invalid Model. Please use one of the following model: {', '.join(Llama2Model.list_values())}",
                error_type="ValueError",
            )
        try:
            api_url = (
                f"https://api-inference.huggingface.co/models/meta-llama/{self.model}"
            )
            headers = {"Authorization": f"Bearer {self.api_key}"}

            def query(payload):
                response = requests.post(api_url, headers=headers, json=payload)
                return response.json()

            # Llama2 prompt template
            prompt_template = f"<s>[INST] <<SYS>>\n{{{{ {self.system_prompt} }}}}\n<</SYS>>\n{{{{ {prompt or self.prompt} }}}}\n[/INST]"

            output = query(
                {
                    "inputs": prompt_template,
                    "parameters": {
                        "max_length": self.max_output_tokens,
                        "temperature": self.temperature,
                        "max_time": self.timeout,
                    },
                }
            )

        except Exception as e:
            raise Llama2Exception(exception=e.args[0], error_type="Llama2Error") from e

        if output["error"]:
            raise Llama2Exception(exception=output["error"], error_type="Llama2Error")

        return output[0]["generated_text"]

    def get_estimated_max_cost(
        self, prompt: str = "", price_data: Dict[str, Any] = None
    ) -> float:
        if not self.prompt and not prompt:
            logger.info("No prompt provided.")
            raise ValueError("No prompt provided.")

        # Assumption, model exists (check should be done at yml load level)
        logger.info("Tokenizing model: %s", self.model)

        prompt_cost_per_token = price_data["prompt"]
        logger.info("Prompt Cost per token: %s", prompt_cost_per_token)

        completion_cost_per_token = price_data["completion"]
        logger.info("Output cost per token: %s", completion_cost_per_token)

        tokens = tokenizer.bpe_tokenize_encode(prompt or self.prompt)

        logger.info("Number of input tokens found: %d", len(tokens))

        logger.info(
            "Final calculation using %d input tokens and %d output tokens",
            len(tokens),
            price_data["max-output-tokens"],
        )

        cost = round(
            prompt_cost_per_token * len(tokens)
            + completion_cost_per_token * price_data["max-output-tokens"],
            8,
        )

        logger.info("Calculated Cost: %s", cost)

        return cost

    def get_category_rank(self, category: str = "") -> str:
        logger.info(msg=f"Current model: {self.model}")
        logger.info(msg=f"Category of prompt: {category}")
        category_rank = llama2_category_data["model-categories"][self.model][category]
        logger.info(msg=f"Rank of category: {category_rank}")
        return category_rank
