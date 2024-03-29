import os
import time

from dotenv import load_dotenv
from proxyllm.provider.openai.chatgpt import OpenAIAdapter
from proxyllm.proxyllm import LLMProxy

load_dotenv(".env.test")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def call_models(prompt: str) -> dict:
    start_openai = time.perf_counter()
    openai = OpenAIAdapter(prompt=prompt, model="gpt-4", api_key=OPENAI_API_KEY)
    openai_output = openai.get_completion(prompt=prompt)
    end_openai = time.perf_counter()
    openai_latency = end_openai - start_openai

    start_llmproxy = time.perf_counter()
    llmproxy = LLMProxy(route_type="cost")
    llmproxy_output = llmproxy.route(prompt=prompt)
    end_llmproxy = time.perf_counter()
    llmproxy_latency = end_llmproxy - start_llmproxy

    return {
        "openai":{
            "response":openai_output,
            "latency":openai_latency
        },
        "llmproxy":{
            "response":llmproxy_output,
            "latency":llmproxy_latency
        }
    }
