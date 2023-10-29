import tiktoken


models_costs = {
    "gpt-3.5-turbo-4k": {
        "prompt": 0.0015,
        "completion": 0.002,
    },
    "gpt-3.5-turbo-16k": {
        "prompt": 0.003,
        "completion": 0.004,
    },
    "gpt-4-8k": {
        "prompt": 0.03,
        "completion": 0.06,
    },
    "gpt-4-32k": {
        "prompt": 0.06,
        "completion": 0.12,
    },
    "text-embedding-ada-002-v2": {
        "prompt": 0.0001,
        "completion": 0.0001,
    },
}


def get_estimated_max_cost(prompt, model, max_output_tokens):
    enc = tiktoken.encoding_for_model(model)

    prompt_cost_per_token = 0.0015 / 1000
    completion_cost_per_token = 0.0015 / 1000
    tokens = enc.encode(prompt)
    print(len(tokens))

    return (
        prompt_cost_per_token * len(tokens)
        + completion_cost_per_token * max_output_tokens
    )


# prompt = "I AM A MAN, I AM A HUMAN, I AM A CAT"
# cost = get_estimated_max_cost(
#     prompt=prompt, model="gpt-3.5-turbo-4k", max_output_tokens=256
# )
# print(cost)
