from transformers import pipeline

class CategoryModel:
    def __init__(self, prompt: str = "") -> None:
        self.prompt = prompt
        self.model = "facebook/bart-large-mnli"

    def categorize_text(self) -> str:
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
        classifier = pipeline(
            task="zero-shot-classification", model=self.model
        )

        results = classifier(self.prompt, candidate_labels)
        best_category = results["labels"][0]

        return best_category