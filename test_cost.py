import time

from proxyllm import LLMProxy

# NOTE: Temp Test file, will be removed in future in favour of unit/integration tests


def main() -> None:
    prompt = "Say 'test'"
    proxy_client = LLMProxy(route_type="cost")
    start = time.time()
    output = proxy_client.route(prompt=prompt)
    end = time.time()

    print("RESPONSE MODEL: ", output.response_model)
    print("RESPONSE: ", output.response)
    print(f"\nCost route total time taken: {end-start}")

    assert output.response_model
    assert output.response


if __name__ == "__main__":
    main()
