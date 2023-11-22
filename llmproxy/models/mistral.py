from llmproxy.models.base import BaseModel, CompletionResponse
from llmproxy.utils.enums import BaseEnum
import requests
from llmproxy.utils.log import logger
from llmproxy.utils import tokenizer

# prices not accurate
mistral_price_data = {
    "max-output-tokens": 50,
    "model-costs": {
        "prompt": 1.30 / 1_000_000,
        "completion": 1.70 / 1_000_000,
    },
}


class MistralModel(str, BaseEnum):
    Mistral_7B = "Mistral-7B-v0.1"
    Mistral_7B_Instruct = "Mistral-7B-Instruct-v0.1"


class Mistral(BaseModel):
    def __init__(
        self,
        prompt: str = "",
        model: MistralModel = MistralModel.Mistral_7B_Instruct.value,
        api_key: str = "",
        temperature: float = 1.0,
        max_output_tokens: int = None,
    ) -> None:
        self.prompt = prompt
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def get_completion(self, prompt: str = "") -> CompletionResponse:
        if self.model not in MistralModel:
            raise MistralException(
                exception=f"Model not supported, please use one of the following: {', '.join(MistralModel.list_values())}",
                error_type=ValueError,
            )
        try:
            API_URL = (
                f"https://api-inference.huggingface.co/models/mistralai/{self.model}"
            )
            headers = {"Authorization": f"Bearer {self.api_key}"}

            def query(payload):
                response = requests.post(API_URL, headers=headers, json=payload)
                return response.json()

            output = query(
                {
                    "inputs": prompt or self.prompt,
                    "parameters": {
                        "temperature": self.temperature,
                        "max_length": self.max_output_tokens,
                    },
                }
            )
        except requests.RequestException as e:
            raise MistralException(f"Request error: {e}", error_type="RequestError")
        except Exception as e:
            raise MistralException(f"Unknown error: {e}", error_type="UnknownError")

        response = ""
        message = ""
        if isinstance(output, list) and "generated_text" in output[0]:
            response = output[0]["generated_text"]
        elif "error" in output:
            raise MistralException(f"{output['error']}", error_type="MistralError")
        else:
            raise ValueError("Unknown output format")

        return CompletionResponse(
            payload=response,
            message=message,
            err="",
        )

    def get_estimated_max_cost(self, prompt: str = "") -> float:
        if not self.prompt and not prompt:
            logger.info("No prompt provided.")
            raise ValueError("No prompt provided.")

        # Assumption, model exists (check should be done at yml load level)
        logger.info(f"Tokenizing model: {self.model}")

        prompt_cost_per_token = mistral_price_data["model-costs"]["prompt"]
        logger.info(f"Prompt Cost per token: {prompt_cost_per_token}")

        completion_cost_per_token = mistral_price_data["model-costs"]["completion"]
        logger.info(f"Output cost per token: {completion_cost_per_token}")

        tokens = tokenizer.bpe_tokenize_encode(prompt or self.prompt)

        logger.info(f"Number of input tokens found: {len(tokens)}")

        logger.info(
            f"Final calculation using {len(tokens)} input tokens and {mistral_price_data['max-output-tokens']} output tokens"
        )

        cost = round(
            prompt_cost_per_token * len(tokens)
            + completion_cost_per_token * mistral_price_data["max-output-tokens"],
            8,
        )

        logger.info(f"Calculated Cost: {cost}")

        return cost


class MistralException(Exception):
    def __init__(self, exception: str, error_type: str) -> None:
        super().__init__(f"Mistral Error: {exception}, Type: {error_type}")
