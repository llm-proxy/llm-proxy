from llmproxy import llmproxy
from llmproxy.utils.log import logger
from llmproxy.models.cohere import CohereModel
from llmproxy.llmproxy import LLMProxy


""" Temp Test file, will be removed in future in favour of unit/integration tests"""


def main() -> None:
    llmproxy = LLMProxy()

    llmproxy.route()

    # print(llmproxy.prompt("test"))

    # prompt = "What is 1+1?"

    # print(f"OPEN AI: {llmproxy.get_completion_openai(prompt=prompt)}")
    # print(f"MISTRAL AI: {llmproxy.get_completion_mistral(prompt=prompt)}")
    # print(f"LLAMA2 AI: {llmproxy.get_completion_llama2(prompt=prompt)}")
    # print(f"COHERE AI: {llmproxy.get_completion_cohere(prompt=prompt)}")
    # print(f"VERTEX AI: {llmproxy.get_completion_vertexai(prompt=prompt,)}")


if __name__ == "__main__":
    main()
