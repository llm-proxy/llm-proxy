import time

from llmproxy.llmproxy import LLMProxy

# NOTE: Temp Test file, will be removed in future in favour of unit/integration tests


def main() -> None:
    # prompt = "I am a man, not a man, but not a man, that is an apple, or a banana!"
    prompt = "what is 1 + 1 equal to?"
    # output = proxy_client.route(route_type="cost", prompt=prompt)
    # print(output)

    start = time.time()
    proxy_client = LLMProxy()
    output = proxy_client.route(prompt=prompt)
    end = time.time()
    print("RESPONSE MODEL: ", output.response_model)
    print("RESPONSE: ", output.response)
    # print("ERRORS: ", output.errors)
    print(f"\Cost route total time taken: {end-start}")

    prompt = "ok take that numerical answer and add the number 3 to it. What is this equal to mathematically?"
    output = proxy_client.route(prompt=prompt)
    print("RESPONSE MODEL: ", output.response_model)
    print("RESPONSE: ", output.response)
    print("ERRORS: ", output.errors)

    proxy_client.clear_chat()

    # prompt = "what is the mathematical equation '2 + 3' equal to?"
    # output = proxy_client.route(prompt=prompt)
    # print("RESPONSE MODEL: ", output.response_model)
    # print("RESPONSE: ", output.response)
    # print("ERRORS: ", output.errors)

    assert output.response_model
    assert output.response


if __name__ == "__main__":
    main()
