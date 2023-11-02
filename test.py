from llmproxy import llmproxy
from llmproxy.utils.log import logger
from llmproxy.models.cohere import CohereModel


def main() -> None:
    prompt = "What is 1+1?"

    print(f"OPEN AI: {llmproxy.get_completion_openai(prompt=prompt)}")
    print(f"MISTRAL AI: {llmproxy.get_completion_mistral(prompt=prompt)}")
    print(f"LLAMA2 AI: {llmproxy.get_completion_llama2(prompt=prompt)}")
    print(f"COHERE AI: {llmproxy.get_completion_cohere(prompt=prompt)}")
    print(f"VERTEX AI: {llmproxy.get_completion_vertexai(prompt=prompt,)}")


if __name__ == "__main__":
    main()
