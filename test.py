from llmproxy import llmproxy


def main() -> None:
    prompt = "What is 1+1 equal to?"
    
    print("OPEN AI: %s", llmproxy.get_completion(prompt=prompt))
    print("MISTRAL AI: %s", llmproxy.get_completion_mistral(prompt=prompt))

if __name__ == "__main__":
    main()
