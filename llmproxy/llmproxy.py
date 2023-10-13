import operator
import os


from llmproxy.models import openai, mistral, cohereai, llama2, base

from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
mistral_api_key = os.getenv("MISTRAL_API_KEY")
cohere_api_key = os.getenv("COHERE_API_KEY")
llama2_api_key = os.getenv("LLAMA2_API_KEY")


def get_available_models(prompt: str) -> dict[str, base.BaseChatbot]:
    models = {}
    if openai_api_key:
        models["openai"] = openai.OpenAI(prompt=prompt, api_key=openai_api_key)
    if mistral_api_key:
        models["mistral"] = mistral.Mistral(prompt=prompt, api_key=mistral_api_key)
    if cohere_api_key:
        models["cohere"] = cohereai.Cohere(message=prompt, api_key=cohere_api_key)
    if llama2_api_key:
        models["llama2"] = llama2.Llama2(prompt=prompt, api_key=llama2_api_key)
    return models


def choose_best_model_with_cost(models: dict[str, base.BaseChatbot]) -> str:
    sorted_models = sorted(
        models.items(),
        key=lambda x: x[1].INPUT_COST_PER_TOKEN + x[1].OUTPUT_COST_PER_TOKEN,
    )

    res = ""
    for model in sorted_models:
        res = model[1].get_completion()
        if res.err:
            continue
        res = f"{model[0]} output:  {res.payload}"
        break
    return res
