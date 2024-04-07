import time

from proxyllm import LLMProxy

# NOTE: Temp Test file, will be removed in future in favour of unit/integration tests


def main() -> None:
    # prompt = "I am a man, not a man, but not a man, that is an apple, or a banana!"
    # prompt = "The quick brown fox jumps over the lazy dog."
    # output = proxy_client.route(route_type="cost", prompt=prompt)
    # print(output)
    # start = time.time()
    # end = time.time()
    # print(f"\nCost route total time taken: {end-start}")

    prompt = "what is 1 + 1"

    proxy_client = LLMProxy(route_type="cost")
    output = proxy_client.route(prompt=prompt)
    chat_history = output.chat_history

    print("RESPONSE MODEL: ", output.response_model)
    print("RESPONSE: ", output.response)
    print("CHAT HISTORY: ", chat_history)

    assert output.response_model
    assert output.response

    # Round 2... FIGHT!
    prompt2 = "ok take that and now add 3, whats the total now?"
    output = proxy_client.route(prompt=prompt2, chat_history=chat_history)
    chat_history = output.chat_history

    print("RESPONSE MODEL: ", output.response_model)
    print("RESPONSE: ", output.response)
    print("CHAT HISTORY: ", chat_history)

    assert output.response_model
    assert output.response


if __name__ == "__main__":
    main()
