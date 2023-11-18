from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class CategoryModel:
    def __init__(self, prompt:str="") -> None:
        self.prompt=prompt
        self.model=AutoModelForSequenceClassification.from_pretrained("llmproxy/data/classification")
        self.tokenizer=AutoTokenizer.from_pretrained('distilbert-base-uncased')
        
    def categorize_text(self) -> str:
        category_names = ["Code Generation Task", "Text Generation Task", "Translation and Multilingual Applications Task", 
                          "Natural Language Processing Task", "Conversational AI Task", "Educational Applications Task",
                          "Healthcare and Medical Task", "Legal Task", "Financial Task", "Content Recommendation Task"]
        tokenized_prompt = self.tokenizer(self.prompt, return_tensors='pt')
        output = self.model(**tokenized_prompt)
        logits = output.logits
        predicted_index = torch.argmax(logits, dim=-1).item()
        predicted_category = category_names[predicted_index]
        return predicted_category