# Import OpenAI's GPT 3.5 Turbo 
import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


def classify_prompt(prompt:str)->str:
    
    openai.api_key=openai_api_key

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": """You are a prompt classification assistant who will classify the user content 
            into the following categories and would ouput only the category name, no other text: Code Generation Task, Text 
            Generation Task, Translation and Multilingual Applications Task, Natural Language Processing Task, Conversational
            AI Task, Educational Applications Task, Healthcare and Medical Task, Legal Task, Financial Task, Content Recommendation Task"""},
            {"role": "user", "content": prompt},
        ]
    )
    
    return response['choices'][0]['message']['content']