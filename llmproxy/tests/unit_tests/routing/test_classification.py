from llmproxy.routing.classification import classify_input

def test_classification_science() -> None:
    prompt = "what is a cell made from?"
    output = classify_input(prompt)
    assert output == "Science"