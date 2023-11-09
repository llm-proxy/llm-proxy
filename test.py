from llmproxy.llmproxy import LLMProxy

from llmproxy import llmproxy

# from llmproxy.utils.log import logger
# from llmproxy.models.cohere import CohereModel

"""Temp Test file, will be removed in future in favour of unit/integration tests"""


def main() -> None:
    proxy_client = LLMProxy()

    proxy_client.route(route_type="cost")

    print(f"OPEN AI FINAL PRICE: {llmproxy.min_cost_openai()}")
    print(f"COHERE AI FINAL PRICE: {llmproxy.min_cost_cohere()}")
    print(f"LLAMA2 AI FINAL PRICE: {llmproxy.min_cost_llama2()}")
    print(f"MISTRAL AI FINAL PRICE: {llmproxy.min_cost_mistral()}")
    print(f"VERTEXAI FINAL PRICE: {llmproxy.min_cost_vertexai()}")

    # # print(f"OPEN AI: {llmproxy.get_completion_openai(prompt=prompt)}")
    # # print(f"MISTRAL AI: {llmproxy.get_completion_mistral(prompt=prompt)}")
    # # print(f"LLAMA2 AI: {llmproxy.get_completion_llama2(prompt=prompt)}")
    # # print(f"COHERE AI: {llmproxy.get_completion_cohere(prompt=prompt)}")
    # # print(f"VERTEX AI: {llmproxy.get_completion_vertexai(prompt=prompt,)}")


if __name__ == "__main__":
    main()
