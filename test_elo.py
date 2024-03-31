import time

from proxyllm import LLMProxy

# NOTE: Temp Test file, will be removed in future in favour of unit/integration tests


def main() -> None:
    # prompt = "I am a man, not a man, but not a man, that is an apple, or a banana!"
    prompt = "what is 1+1?"
    # output = proxy_client.route(route_type="cost", prompt=prompt)
    # print(output)

    start = time.time()
    proxy_client = LLMProxy(route_type="elo")
    output = proxy_client.route(prompt=prompt)
    end = time.time()
    print("RESPONSE MODEL: ", output.response_model)
    print("RESPONSE: ", output.response)
    # print("ERRORS: ", output.errors)
    print(f"\nCategory route total time taken: {end-start}")
    assert output.response_model
    assert output.response


if __name__ == "__main__":
    main()
