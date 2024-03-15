from typing import Any, Dict


def calculate_estimated_max_cost(
    price_data: Dict[str, Any],
    num_of_input_tokens: int = 0,
    max_output_tokens: int = 256,
) -> float:
    """
    Calculate the estimated maximum cost for generating completion output based on provided price data.

    Args:
        price_data (Dict[str, Any]): A dictionary containing the price data, including the cost per token for prompt
            and completion generation.
        num_of_input_tokens (int, optional): The number of tokens in the prompt input. Defaults to 0.
        max_output_tokens (int, optional): The maximum number of tokens in the completion output. Defaults to 256.

    Returns:
        float: The estimated maximum cost for generating completion output.

    Raises:
        KeyError: If the 'prompt' or 'completion' key is missing in the price_data dictionary.

    Example:
        >>> price_data = {"prompt": 0.02, "completion": 0.03}
        >>> calculate_estimated_max_cost(price_data, 100, 300)
        12.0
    """
    prompt_cost_per_token = price_data["prompt"]
    completion_cost_per_token = price_data["completion"]

    cost = round(
        prompt_cost_per_token * num_of_input_tokens
        + completion_cost_per_token * max_output_tokens,
        8,
    )

    return cost
