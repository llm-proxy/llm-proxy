from src import llmproxy


prompt = "What is 1+1?"

print(llmproxy.getCompletion(prompt=prompt))
