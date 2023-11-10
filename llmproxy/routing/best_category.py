from dotenv import load_dotenv
from transformers import pipeline
from llmproxy.models.openai import OpenAI
from llmproxy.models.cohere import Cohere 
from llmproxy.models.mistral import Mistral
from llmproxy.models.llama2 import Llama2
from llmproxy.models.vertexai import VertexAI
from typing import List, Union
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

    #print(f'Prompt = {prompt}')
    classifier = pipeline(task='zero-shot-classification', model='facebook/bart-large-mnli')
    
    results = classifier(prompt, candidate_labels)
    #print(f'labels = {results["labels"]}')
    #print(f'scores = {results["scores"]}')

    best_category = results['labels'][0]

    return best_category


def use_best_category(prompt:str) -> List[Union(OpenAI, Cohere, Llama2, Mistral, VertexAI)]:
    category = classify_input(prompt)
    print(category)
    if(category=="Code Generation Task" or category=="Text Generation Task" or category=="Natural Language Processing Task"):
            return [OpenAI, Cohere, VertexAI, Mistral, Llama2]
    elif(category=="Conversational AI Task" or category=="Educational Applications Task" or category=="Healthcare and Medical Task"
          or category=="Legal Task" or category=="Financial Task" or category=="Content Recommendation Task"):
         return [OpenAI, Llama2, Cohere, VertexAI]
    elif(category=="Translation and Multilingual Applications Task"):
         return[OpenAI, Llama2, Cohere]
     
if __name__ == "__main__":
    model_list = use_best_category('What are the effects marijuana?')
    print(model_list)