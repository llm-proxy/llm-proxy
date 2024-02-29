import json
import os

from tokenizers import Encoding, Tokenizer

"""" Model current does not handle all special tokens: ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]"""


def bpe_tokenize_encode(prompt: str = "") -> Encoding:
    # Get the absolute path to the directory containing the script
    script_dir = os.path.abspath(os.path.dirname(__file__))
    # Construct the absolute path to the JSON file
    json_file_path = os.path.join(script_dir, "../data/tokenizer-wiki.json")
    tokenizer = Tokenizer.from_file(json_file_path)
    return tokenizer.encode(prompt)


def vertexai_encode(prompt: str = "") -> str:
    return prompt.replace(" ", "")
