from src import llmproxy


def main() -> None:
    prompt = "What is 1+1?"
    print(llmproxy.get_completion(prompt=prompt))


if __name__ == "__main__":
    main()
