from llmproxy.routing.classification import ClassificationModel

def test_classification_science() -> None:
    # Arrange 
    prompt = "what is a cell made from?"
    model = ClassificationModel()
    
    # Act 
    output = model.classify_input(prompt)
    
    # Assert
    assert output == "Science"
        