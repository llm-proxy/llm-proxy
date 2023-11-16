import os
from dotenv import load_dotenv
from llmproxy.models.cohere import Cohere

load_dotenv(".env.test")

cohere_api_key = os.getenv("COHERE_API_KEY")


# TODO: May be a FLAKY test; Ensure this is not the case
def test_get_estimated_max_cost():
    cohere = Cohere(
        api_key=cohere_api_key,
        model="command",
        temperature=0,
    )
    estimated_cost = 0.000112

    prompt = "I am a cat in a hat!"
    actual_cost = cohere.get_estimated_max_cost(prompt=prompt)
    assert actual_cost == estimated_cost, "NOTE: Flaky test may need to be changed/removed in future based on pricing"
