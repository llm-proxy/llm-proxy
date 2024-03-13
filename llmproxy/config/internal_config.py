"""
Data for the internal configuration
Used to compare User inputs to supported providers/models

Format: 
    {
        "provider": "Name of provider",
        "adapter_path": "Path to provider specific adapter",
        "models": [
            {
                "name": "name of model",
                "cost_per_token_input": Flat price per output token,
                "cost_per_token_output": Flat price per input token,
            },
        ],
    }
]


"""

from typing import Any, Dict, List

internal_config: List[Dict[str, Any]] = [
    {
        "provider": "OpenAI",
        "adapter_path": "llmproxy.provider.openai.chatgpt.OpenAIAdapter",
        "models": [
            {
                "name": "gpt-3.5-turbo-1106",
                "cost_per_token_input": 1e-06,
                "cost_per_token_output": 2e-06,
            },
            {
                "name": "gpt-3.5-turbo-instruct",
                "cost_per_token_input": 1.5e-06,
                "cost_per_token_output": 2e-06,
            },
            {
                "name": "gpt-4",
                "cost_per_token_input": 3e-05,
                "cost_per_token_output": 6e-05,
            },
            {
                "name": "gpt-4-32k",
                "cost_per_token_input": 6e-05,
                "cost_per_token_output": 0.00012,
            },
        ],
    },
    {
        "provider": "Llama2",
        "adapter_path": "llmproxy.provider.huggingface.llama2.Llama2Adapter",
        "models": [
            {
                "name": "Llama-2-7b-chat-hf",
                "cost_per_token_input": 5e-08,
                "cost_per_token_output": 2.5e-07,
            },
            {
                "name": "Llama-2-7b-chat",
                "cost_per_token_input": 5e-08,
                "cost_per_token_output": 2.5e-07,
            },
            {
                "name": "Llama-2-7b-hf",
                "cost_per_token_input": 5e-08,
                "cost_per_token_output": 2.5e-07,
            },
            {
                "name": "Llama-2-7b",
                "cost_per_token_input": 5e-08,
                "cost_per_token_output": 2.5e-07,
            },
            {
                "name": "Llama-2-13b-chat-hf",
                "cost_per_token_input": 1e-07,
                "cost_per_token_output": 5e-07,
            },
            {
                "name": "Llama-2-13b-chat",
                "cost_per_token_input": 1e-07,
                "cost_per_token_output": 5e-07,
            },
            {
                "name": "Llama-2-13b-hf",
                "cost_per_token_input": 1e-07,
                "cost_per_token_output": 5e-07,
            },
            {
                "name": "Llama-2-13b",
                "cost_per_token_input": 1e-07,
                "cost_per_token_output": 5e-07,
            },
            {
                "name": "Llama-2-70b-chat-hf",
                "cost_per_token_input": 6.5e-07,
                "cost_per_token_output": 2.75e-06,
            },
            {
                "name": "Llama-2-70b-chat",
                "cost_per_token_input": 6.5e-07,
                "cost_per_token_output": 2.75e-06,
            },
            {
                "name": "Llama-2-70b-hf",
                "cost_per_token_input": 6.5e-07,
                "cost_per_token_output": 2.75e-06,
            },
            {
                "name": "Llama-2-70b",
                "cost_per_token_input": 6.5e-07,
                "cost_per_token_output": 2.75e-06,
            },
        ],
    },
    {
        "provider": "Mistral",
        "adapter_path": "llmproxy.provider.huggingface.mistral.MistralAdapter",
        "models": [
            {
                "name": "Mixtral-7B-v0.1",
                "cost_per_token_input": 5e-08,
                "cost_per_token_output": 2.5e-07,
            },
            {
                "name": "Mixtral-7B-Instruct-v0.2",
                "cost_per_token_input": 5e-08,
                "cost_per_token_output": 2.5e-07,
            },
            {
                "name": "Mixtral-8x7B-Instruct-v0.1",
                "cost_per_token_input": 3e-07,
                "cost_per_token_output": 1e-06,
            },
        ],
    },
    {
        "provider": "Cohere",
        "adapter_path": "llmproxy.provider.cohere.cohere.CohereAdapter",
        "models": [
            {
                "name": "command",
                "cost_per_token_input": 1.5e-06,
                "cost_per_token_output": 2e-06,
            },
            {
                "name": "command-light",
                "cost_per_token_input": 1.5e-06,
                "cost_per_token_output": 2e-06,
            },
            {
                "name": "command-nightly",
                "cost_per_token_input": 1.5e-06,
                "cost_per_token_output": 2e-06,
            },
            {
                "name": "command-light-nightly",
                "cost_per_token_input": 1.5e-06,
                "cost_per_token_output": 2e-06,
            },
        ],
    },
    {
        "provider": "Vertexai",
        "adapter_path": "llmproxy.provider.google.vertexai.VertexAIAdapter",
        "models": [
            {
                "name": "text-bison",
                "cost_per_token_input": 1.5e-06,
                "cost_per_token_output": 2e-06,
            }
        ],
    },
]
