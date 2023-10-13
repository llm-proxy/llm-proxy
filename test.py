from llmproxy import llmproxy


def main() -> None:
    prompt = "What is 1+1 equal to?"
    # print("OPEN AI: %s", llmproxy.get_completion(prompt=prompt))
    # print("=======================================================")
    # print("COHERE AI: %s", llmproxy.get_completion_cohere(prompt=prompt))
    # print("=======================================================")
    # print("MISTRAL AI: %s", llmproxy.get_completion_mistral(prompt=prompt))
    # print("=======================================================")
    # print("LLAMA 2 AI: %s", llmproxy.get_completion_llama2(prompt=prompt))

    models = llmproxy.get_available_models(prompt=prompt)
    best_model_response = llmproxy.choose_best_model_with_cost(models=models)

    print(best_model_response)


if __name__ == "__main__":
    main()
