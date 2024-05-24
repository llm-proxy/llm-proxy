"""
Defines internal configuration for mapping user inputs to supported LLM providers and models.
This configuration helps in estimating costs by detailing the per-token price for inputs and outputs across different models.

Format: 
    {
        "provider": "Name of provider",
        "adapter_path": "Path to provider specific adapter",
        "models": [
            {
                "name": "name of model",
                "cost_per_token_input": Flat price per output token,
                "cost_per_token_output": Flat price per input token,
                "elo": elo rating,
            },
        ],
    }
]
"""

from typing import Any, Dict, List

internal_config: List[Dict[str, Any]] = [
    {
        "provider": "OpenAI",
        "adapter_path": "proxyllm.provider.openai.chatgpt.OpenAIAdapter",
        "models": [
            {
                "name": "gpt-4-turbo",
                "cost_per_token_input": 10e-06,
                "cost_per_token_output": 30e-06,
                "elo": 1287,
            },
            {
                "name": "gpt-4o",
                "cost_per_token_input": 5e-06,
                "cost_per_token_output": 15e-06,
                "elo": 1287,
            },
            # GPT-4 Turbo
            {
                "name": "gpt-4-0125-preview",
                "cost_per_token_input": 10e-06,
                "cost_per_token_output": 30e-06,
                "elo": 1248,
            },
            {
                "name": "gpt-4-1106-preview",
                "cost_per_token_input": 10e-06,
                "cost_per_token_output": 30e-06,
                "elo": 1251,
            },
            # GPT-4 (points to gpt-4-0613
            {
                "name": "gpt-4",
                "cost_per_token_input": 30e-06,
                "cost_per_token_output": 60e-06,
                "elo": 1158,
            },
            {
                "name": "gpt-4-32k",
                "cost_per_token_input": 60e-06,
                "cost_per_token_output": 120e-06,
                "elo": 1158,  # assumed elo is the same for gpt-4
            },
            # GPT-3.5 Turbo || LEGACY
            # {
            #     "name": "gpt-3.5-turbo-instruct",
            #     "cost_per_token_input": 1.5e-06,
            #     "cost_per_token_output": 2e-06,
            #     "elo": 0,
            # },
            {
                "name": "gpt-3.5-turbo-0125",
                "cost_per_token_input": 0.5e-06,
                "cost_per_token_output": 1.5e-06,
                "elo": 1097,
            },
            # Older Models
            {
                "name": "gpt-3.5-turbo-1106",
                "cost_per_token_input": 1e-06,
                "cost_per_token_output": 2e-06,
                "elo": 1068,
            },
            {
                "name": "gpt-3.5-turbo-0613",
                "cost_per_token_input": 1.5e-06,
                "cost_per_token_output": 2e-06,
                "elo": 1114,
            },
            {
                "name": "gpt-3.5-turbo-16k-0613",
                "cost_per_token_input": 3e-06,
                "cost_per_token_output": 4e-06,
                "elo": 1114,  # assumed elo is the same for gpt-3.5-turbo-0613
            },
            {
                "name": "gpt-3.5-turbo-0301",
                "cost_per_token_input": 1.5e-06,
                "cost_per_token_output": 2e-06,
                "elo": 499,  # no data for this model
            },
        ],
    },
    {
        "provider": "Llama2",
        "adapter_path": "proxyllm.provider.huggingface.llama2.Llama2Adapter",
        "models": [
            {
                "name": "llama-2-7b-chat-hf",
                "cost_per_token_input": 5e-08,
                "cost_per_token_output": 2.5e-07,
                "elo": 1030,
            },
            {
                "name": "llama-2-7b-chat",
                "cost_per_token_input": 5e-08,
                "cost_per_token_output": 2.5e-07,
                "elo": 1030,
            },
            {
                "name": "llama-2-7b-hf",
                "cost_per_token_input": 5e-08,
                "cost_per_token_output": 2.5e-07,
                "elo": 1030,
            },
            {
                "name": "llama-2-7b",
                "cost_per_token_input": 5e-08,
                "cost_per_token_output": 2.5e-07,
                "elo": 1030,
            },
            {
                "name": "llama-2-13b-chat-hf",
                "cost_per_token_input": 1e-07,
                "cost_per_token_output": 5e-07,
                "elo": 1044,
            },
            {
                "name": "llama-2-13b-chat",
                "cost_per_token_input": 1e-07,
                "cost_per_token_output": 5e-07,
                "elo": 1044,
            },
            {
                "name": "llama-2-13b-hf",
                "cost_per_token_input": 1e-07,
                "cost_per_token_output": 5e-07,
                "elo": 1044,
            },
            {
                "name": "llama-2-13b",
                "cost_per_token_input": 1e-07,
                "cost_per_token_output": 5e-07,
                "elo": 1044,
            },
            {
                "name": "llama-2-70b-chat-hf",
                "cost_per_token_input": 6.5e-07,
                "cost_per_token_output": 2.75e-06,
                "elo": 1082,
            },
            {
                "name": "llama-2-70b-chat",
                "cost_per_token_input": 6.5e-07,
                "cost_per_token_output": 2.75e-06,
                "elo": 1082,
            },
            {
                "name": "llama-2-70b-hf",
                "cost_per_token_input": 6.5e-07,
                "cost_per_token_output": 2.75e-06,
                "elo": 1082,
            },
            {
                "name": "llama-2-70b",
                "cost_per_token_input": 6.5e-07,
                "cost_per_token_output": 2.75e-06,
                "elo": 1082,
            },
        ],
    },
    {
        "provider": "Cohere",
        "adapter_path": "proxyllm.provider.cohere.cohere.CohereAdapter",
        "models": [
            {
                "name": "command-r-plus",
                "cost_per_token_input": 3.0e-06,
                "cost_per_token_output": 15.0e-06,
                "elo": 1188,
            },
            {
                "name": "command-r",
                "cost_per_token_input": 0.5e-06,
                "cost_per_token_output": 1.5e-06,
                "elo": 1120,  # approximated, should have a similar elo rating to mixtral-8x7b-instruct-v0.1
            },
            {
                "name": "command",
                "cost_per_token_input": 1.5e-06,
                "cost_per_token_output": 2e-06,
                "elo": 1050,  # estimated, previous version of command-r
            },
            {
                "name": "command-light",
                "cost_per_token_input": 1.5e-06,
                "cost_per_token_output": 2e-06,
                "elo": 1025,  # estimated, smaller version of command
            },
            {
                "name": "command-nightly",
                "cost_per_token_input": 1.5e-06,
                "cost_per_token_output": 2e-06,
                "elo": 500,  # estimated, latest but unstable
            },
            {
                "name": "command-light-nightly",
                "cost_per_token_input": 1.5e-06,
                "cost_per_token_output": 2e-06,
                "elo": 499,  # estimated, latest but unstable
            },
        ],
    },
    {
        "provider": "Vertexai",
        "adapter_path": "proxyllm.provider.google.vertexai.VertexAIAdapter",
        "models": [
            {
                "name": "text-bison",
                "cost_per_token_input": 1.5e-06,
                "cost_per_token_output": 2e-06,
                "elo": 1003,  # estimated, similar to chat-bison but should rank higher
            },
            {
                "name": "gemini-pro",
                "cost_per_token_input": 1.25e-07,
                "cost_per_token_output": 3.75e-07,
                "elo": 1125,
            },
            {
                "name": "chat-bison",
                "cost_per_token_input": 1.5e-06,
                "cost_per_token_output": 2e-06,
                "elo": 1002,
            },
            {
                "name": "code-bison",
                "cost_per_token_input": 1.5e-07,
                "cost_per_token_output": 2e-07,
                "elo": 999,  # estimated
            },
            {
                "name": "codechat-bison",
                "cost_per_token_input": 1.5e-07,
                "cost_per_token_output": 2e-07,
                "elo": 1000,  # estimated
            },
            {
                "name": "code-gecko",
                "cost_per_token_input": 1.5e-07,
                "cost_per_token_output": 2e-07,
                "elo": 1001,  # estimated
            },
        ],
    },
    {
        "provider": "Anthropic",
        "adapter_path": "proxyllm.provider.anthropic.claude.ClaudeAdapter",
        "models": [
            {
                "name": "claude-3-opus-20240229",
                "cost_per_token_input": 1.5e-05,
                "cost_per_token_output": 7.5e-05,
                "elo": 1255,
            },
            {
                "name": "claude-3-sonnet-20240229",
                "cost_per_token_input": 3e-06,
                "cost_per_token_output": 1.5e-05,
                "elo": 1200,
            },
            {
                "name": "claude-3-haiku-20240307",
                "cost_per_token_input": 2.5e-07,
                "cost_per_token_output": 1.25e-06,
                "elo": 1177,
            },
        ],
    },
    {
        "provider": "Mistral",
        "adapter_path": "proxyllm.provider.mistral.mistral.MistralAdapter",
        "models": [
            {
                "name": "open-mistral-7b",
                "cost_per_token_input": 2.5e-07,
                "cost_per_token_output": 2.5e-07,
                "elo": 1003,
            },
            {
                "name": "open-mixtral-8x7b",
                "cost_per_token_input": 7e-07,
                "cost_per_token_output": 7e-07,
                "elo": 1073,
            },
            {
                "name": "mistral-small-latest",
                "cost_per_token_input": 2e-06,
                "cost_per_token_output": 6e-06,
                "elo": 1114,
            },
            {
                "name": "mistral-medium-latest",
                "cost_per_token_input": 2.7e-06,
                "cost_per_token_output": 8.1e-06,
                "elo": 1114,
            },
            {
                "name": "mistral-large-latest",
                "cost_per_token_input": 8e-06,
                "cost_per_token_output": 2.4e-05,
                "elo": 1114,
            },
        ],
    },
]
