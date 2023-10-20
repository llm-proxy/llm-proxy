from llmproxy import llmproxy


def main() -> None:
    prompt = "What is 1+1?"
    system_prompt = "Answer correctly"
    model= "Llama-2-7b-chat-hf"
    print(llmproxy.get_completion(prompt=prompt))
    print(llmproxy.get_completion_llama2(prompt=prompt,system_prompt=system_prompt,model=model))

if __name__ == "__main__":
    main()
