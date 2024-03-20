import os
import time

from dotenv import load_dotenv

from proxyllm import LLMProxy
from proxyllm.provider.anthropic import claude

# NOTE: Temp Test file, will be removed in future in favour of unit/integration tests


def main() -> None:
    # prompt = "I am a man, not a man, but not a man, that is an apple, or a banana!"
    prompt = "what is 1 + 1"
    prompt = "The quick brown fox jumps over the lazy dog."
    # output = proxy_client.route(route_type="cost", prompt=prompt)
    # print(output)

    start = time.time()
    proxy_client = LLMProxy(route_type="cost")
    output = proxy_client.route(prompt=prompt)
    end = time.time()
    print("RESPONSE MODEL: ", output.response_model)
    print("RESPONSE: ", output.response)
    # print("ERRORS: ", output.errors)
    print(f"\nCost route total time taken: {end-start}")
    assert output.response_model
    assert output.response


def request():
    prompt = "write me a short story about a homeless man in new york"
    load_dotenv(".env")
    client = claude.ClaudeAdapter(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        model="claude-3-haiku-20240307",
        max_output_tokens=256,
        temperature=0,
    )
    print(client.get_completion(prompt))


if __name__ == "__main__":
    main()
