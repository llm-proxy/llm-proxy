from llmproxy.llmproxy import LLMProxy

from llmproxy import llmproxy

# from llmproxy.utils.log import logger
# from llmproxy.models.cohere import CohereModel

"""Temp Test file, will be removed in future in favour of unit/integration tests"""


def main() -> None:
    # prompt = "I am a man, not a man, but not a man, that is an apple, or a banana!"
    prompt = "What is the square root of i"
    proxy_client = LLMProxy(path_to_configuration="api_configuration.yml")

    output = proxy_client.route(route_type="cost", prompt=prompt)
    print(output)

    # print(f"COHERE AI FINAL PRICE: {llmproxy.min_cost_cohere_example()}")
    # print(f"COHERE AI: {llmproxy.get_completion_cohere_example(prompt=prompt)}")


if __name__ == "__main__":
    main()
