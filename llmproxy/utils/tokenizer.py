from tokenizers import Encoding, Tokenizer

"""" Model current does not handle all special tokens: ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]"""


def bpe_tokenize_encode(prompt: str = "") -> Encoding:
    tokenizer = Tokenizer.from_file("llmproxy/data/tokenizer-wiki.json")
    return tokenizer.encode(prompt)


def vertexai_encode(prompt: str = "") -> str:
    return prompt.replace(" ", "")
