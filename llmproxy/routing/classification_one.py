from transformers import pipeline
from llmproxy.models.openai import OpenAI
from llmproxy.models.cohere import Cohere 
from llmproxy.models.mistral import Mistral
from llmproxy.models.llama2 import Llama2
from llmproxy.models.vertexai import VertexAI
from typing import List, Union
import os
import requests

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
    
    classifier = pipeline(task='zero-shot-classification', model='facebook/bart-large-mnli')
    results = classifier(prompt, candidate_labels)
    best_category = results['labels'][0]

    return best_category


def llm_list(prompt:str) -> List[Union[OpenAI, Cohere, Llama2, Mistral, VertexAI]]:
    category = classify_input(prompt)
    
    if(category=="Code Generation Task" or category=="Text Generation Task" or category=="Natural Language Processing Task"):
            return [OpenAI, Cohere, VertexAI, Mistral, Llama2]
    elif(category=="Conversational AI Task" or category=="Educational Applications Task" or category=="Healthcare and Medical Task"
          or category=="Legal Task" or category=="Financial Task" or category=="Content Recommendation Task"):
         return [OpenAI, Llama2, Cohere, VertexAI]
    elif(category=="Translation and Multilingual Applications Task"):
         return[OpenAI, Llama2, Cohere]