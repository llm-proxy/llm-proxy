from llmproxy import llmproxy


def main() -> None:
    prompt = "What is 1+1?"
    print(llmproxy.get_completion(prompt=prompt))
    print("")
    print(llmproxy.textGenerate(prompt=prompt))


if __name__ == "__main__":
    main()
