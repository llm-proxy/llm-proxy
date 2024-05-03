import time

from proxyllm import LLMProxy

# NOTE: Temp Test file, will be removed in future in favour of unit/integration tests


def main() -> None:
    prompt = "what is 1+1"
    chat_history = [
        {"role": "system", "content": "you are math bot, reply short and sweet"},
    ]
    proxy_client = LLMProxy(route_type="cost")
    start = time.time()
    output = proxy_client.route(prompt=prompt, chat_history=chat_history)
    end = time.time()
    chat_history = output.chat_history

    print("RESPONSE MODEL: ", output.response_model)
    print("RESPONSE: ", output.response)
    print(f"\nCost route total time taken: {end-start}")
    print("CHAT HISTORY: ", chat_history)

    assert output.response_model
    assert output.response

    # Round 2... FIGHT!

    prompt2 = "What was the first question that I asked you?"
    start = time.time()
    output = proxy_client.route(prompt=prompt2, chat_history=chat_history)
    end = time.time()
    chat_history = output.chat_history

    print("RESPONSE MODEL: ", output.response_model)
    print("RESPONSE: ", output.response)
    print(f"\nCost route total time taken: {end-start}")
    print("CHAT HISTORY: ", chat_history)

    assert output.response_model
    assert output.response


if __name__ == "__main__":
    main()
