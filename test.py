from llmproxy import llmproxy
from llmproxy.utils.log import logger

""" Temp Test file, will be removed in future in favour of unit/integration tests"""


def main() -> None:
    prompt = "What is 1+1?"

    print(f"OPEN AI: {llmproxy.get_completion_openai(prompt=prompt)}")
    print(f"MISTRAL AI: {llmproxy.get_completion_mistral(prompt=prompt)}")
    print(f"COHERE AI: {llmproxy.get_completion_cohere(prompt=prompt)}")
    print(f"LLAMA2: {llmproxy.get_completion_llama2(prompt=prompt)}")
    print(
        f"VERTEX AI: {llmproxy.get_completion_vertexai(prompt=prompt,location='us-central1')}"
    )


if __name__ == "__main__":
    main()
