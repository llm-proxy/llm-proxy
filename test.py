from llmproxy.llmproxy import LLMProxy

"""Temp Test file, will be removed in future in favour of unit/integration tests"""


def main() -> None:
    # prompt = "I am a man, not a man, but not a man, that is an apple, or a banana!"
    prompt = "what is 1+1?"
    # output = proxy_client.route(route_type="cost", prompt=prompt)
    # print(output)
    proxy_client = LLMProxy(
        path_to_user_configuration="api_configuration.yml",
        path_to_dev_configuration="developer_configuration.yml",
    )
    output = proxy_client.route(route_type="category", prompt=prompt)
    print(output.payload)


if __name__ == "__main__":
    main()
