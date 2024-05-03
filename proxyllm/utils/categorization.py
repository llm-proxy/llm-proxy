from transformers import pipeline

from proxyllm.utils import proxy_logger


def categorize_text(prompt: str) -> str:
    """
    Categorizes the given text prompt into predefined categories using a zero-shot classification approach.

    Utilizes the 'facebook/bart-large-mnli' model to classify the prompt into one of several predefined task categories.
    This function is intended to aid in determining the most appropriate model for handling a given prompt based on
    its content.

    Args:
        prompt (str): The text prompt to be categorized.

    Returns:
        str: The category that best fits the prompt from the predefined set.
    """
    model = "facebook/bart-large-mnli"
    candidate_labels = [
        "Code Generation Task",
        "Text Generation Task",
        "Translation and Multilingual Applications Task",
        "Natural Language Processing Task",
        "Conversational AI Task",
        "Educational Applications Task",
        "Healthcare and Medical Task",
        "Legal Task",
        "Financial Task",
        "Content Recommendation Task",
    ]
    proxy_logger.log(msg="Classification model is classifying the user prompt")
    classifier = pipeline(task="zero-shot-classification", model=model)
    proxy_logger.log(msg="The prompt has been classified\n")

    results = classifier(prompt, candidate_labels)
    best_category = results["labels"][0]
    return best_category
