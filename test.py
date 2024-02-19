from llmproxy.llmproxy import LLMProxy

# NOTE: Temp Test file, will be removed in future in favour of unit/integration tests


def category_routing() -> None:
    # prompt = "I am a man, not a man, but not a man, that is an apple, or a banana!"
    prompt = "what is 1+1?"
    # output = proxy_client.route(route_type="cost", prompt=prompt)
    # print(output)

    proxy_client = LLMProxy(route_type="category")
    output = proxy_client.route(prompt=prompt)
    print("RESPONSE MODEL: ", output.response_model)
    print("RESPONSE: ", output.response)
    print("ERRORS: ", output.errors)


def cost_routing() -> None:
    # prompt = "I am a man, not a man, but not a man, that is an apple, or a banana!"
    prompt = "what is 1+1?"
    # output = proxy_client.route(route_type="cost", prompt=prompt)
    # print(output)

    proxy_client = LLMProxy(route_type="cost")
    output = proxy_client.route(prompt=prompt)
    print("RESPONSE MODEL: ", output.response_model)
    print("RESPONSE: ", output.response)
    print("ERRORS: ", output.errors)


def main() -> None:
    cost_routing()
    category_routing()


if __name__ == "__main__":
    main()
