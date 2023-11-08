from tokenizers import Tokenizer, Encoding


def tokenize(prompt: str = "") -> Encoding:
    tokenizer = Tokenizer.from_file("llmproxy/data/tokenizer-wiki.json")
    return tokenizer.encode(prompt)


op = tokenize("Hello, y'all! How are you 😁 ?")
print(op.tokens)
