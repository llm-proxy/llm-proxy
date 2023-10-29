from llmproxy import llmproxy
from llmproxy.utils.log import logger


def main() -> None:
    # prompt = "What is 1+1?"
    # print(llmproxy.get_completion(prompt=prompt))

    # print(f"OPEN AI: {llmproxy.get_completion(prompt=prompt)}\n")
    # print(f"MISTRAL AI: {llmproxy.get_completion_mistral(prompt=prompt, model='test')}")

    llmproxy.min_cost()


if __name__ == "__main__":
    main()
