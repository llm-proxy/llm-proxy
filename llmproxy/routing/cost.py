from transformers import AutoModelForCausalLM, AutoTokenizer

# This is super slow and creates a very heavy load
model = AutoModelForCausalLM.from_pretrained("bert-base-cased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

res = tokenizer.tokenize("I AM A HUMAN!@")
print(res)
# OP from "mistralai": ['▁I', '▁AM', '▁A', '▁H', 'UM', 'AN', '!', '@']
