from src import llmproxy

prompt = "What is 1+1?"
prompt2 = "What is an LLM?"

print(llmproxy.getCompletion(prompt=prompt))
print(llmproxy.getVertexCompletion(prompt=prompt2))
