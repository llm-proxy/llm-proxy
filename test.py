from llmproxy import llmproxy


def main() -> None:
    prompt = "What is 1+1 equal to?"

    print(f"OPEN AI: {llmproxy.get_completion(prompt=prompt)}\n")
    print(f"MISTRAL AI: {llmproxy.get_completion_mistral(prompt=prompt, model='test')}\n")

if __name__ == "__main__":
    main()
