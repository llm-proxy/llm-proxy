import json
import os

from proxyllm.proxyllm import LLMProxy


def llmproxy_call(prompt):
    api_key = os.environ.get("OPENAI_API_KEY")
    client = LLMProxy(api_key=api_key, route_type="cost")
    output = client.route(prompt=prompt)
    return output


def function(request):
    req_data = request.get_json()

    prompt_data = req_data.get("prompt", None)
    print(prompt_data)
    if prompt_data is not None:
        if isinstance(prompt_data, str):
            prompt = prompt_data
        elif isinstance(prompt_data, dict):
            prompt = prompt_data.get("text", "")
            if not isinstance(prompt, str):
                return "Invalid prompt format", 400
        else:
            return "Invalid prompt format", 400

        prompt = str(prompt)

        prompt = prompt.encode("utf-8")

        response = llmproxy_call(prompt)

        return json.dumps(response), 200, {"Content-Type": "application/json"}
    else:
        return "Prompt is missing", 400
