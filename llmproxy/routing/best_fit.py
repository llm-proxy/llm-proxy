from transformers import pipeline

def get_best_category(prompt: str, candidate_labels: list[str]) -> str:
    classifier = pipeline(task="zero-shot-classification", model="facebook/bart-large-mnli")
    res = classifier(prompt, candidate_labels=candidate_labels)

    best_category = res["labels"][res["scores"].index(max(res["scores"]))]
    return res, best_category


# Testing
candidate_labels=["business", "math", "code to language", "language to code", "code generation"]
prompt = "Please tell me, what are the top 10 stocks"
_, best_category = get_best_category(prompt=prompt, candidate_labels=candidate_labels)
print(best_category)

prompt = "I want to write python code that converts even numbers to odd and multiplies by 2"
res, best_category = get_best_category(prompt=prompt, candidate_labels=candidate_labels)
print(res)
print(best_category)