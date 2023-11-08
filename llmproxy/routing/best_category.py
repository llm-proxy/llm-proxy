from dotenv import load_dotenv
from transformers import pipeline

import os
import requests


# load_dotenv(".env")
# huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")

# API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
# headers = {"Authorization": f"Bearer {huggingface_api_key}"}

# def query(payload):
# 	response = requests.post(API_URL, headers=headers, json=payload)
# 	return response.json()

# output = query({
#     "inputs": "Hi, I recently bought a device from your company but it is not working as advertised and I would like to get reimbursed!",
#     "parameters": {"candidate_labels": ["refund", "legal", "faq"]},
# })

def classify_input(prompt:str) -> str:
    candidate_labels = [
        'Code Generation Task',
        'Text Generation Task',
        'Translation and Multilingual Applications Task',
        'Natural Language Processing Task',
        'Conversational AI Task',
        'Educational Applications Task',
        'Healthcare and Medical Task',
        'Legal Task',
        'Financial Task',
        'Content Recommendation Task',
    ]

    print(f'Prompt = {prompt}')
    classifier = pipeline(task='zero-shot-classification', model='facebook/bart-large-mnli')
    
    results = classifier(prompt, candidate_labels)
    print(f'labels = {results["labels"]}')
    print(f'scores = {results["scores"]}')

    best_category = results['labels'][0]

    return best_category

print(classify_input('What are the effects marijuana?'))

def use_best_category(best_category:str):
    pass