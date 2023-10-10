from llmproxy import llmproxy


def main() -> None:
    prompt = "What is 1+1?"
    print("OPEN AI: %s", llmproxy.get_completion(prompt=prompt))
    print("COHERE AI: %s", llmproxy.get_completion_cohere(prompt=prompt))

if __name__ == "__main__":
    main()
