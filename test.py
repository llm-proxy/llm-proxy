from llmproxy import llmproxy
from llmproxy.utils.log import logger


def main() -> None:
    prompt = "What is 1+1?"
    logger.info(llmproxy.get_completion(prompt=prompt))


if __name__ == "__main__":
    main()
