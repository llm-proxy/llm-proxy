from llmproxy.routing.classification import ClassificationModel

def test_classification_science() -> None:
    prompt = "what is a cell made from?"
    model = ClassificationModel()
    output = model.classify_input(prompt)
    print(f"{output} == Science")
        