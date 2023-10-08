import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('MISTRAL_API')

API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-v0.1"
headers = {"Authorization": f"Bearer {api_key}"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
output = query({
	"inputs": "Canada is",
})

# print(output[0]['generated_text'])