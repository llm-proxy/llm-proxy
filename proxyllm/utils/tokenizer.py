import os

from tokenizers import Encoding, Tokenizer

"""" Model current does not handle all special tokens: ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]"""


def bpe_tokenize_encode(prompt: str = "") -> Encoding:
    """
    Tokenizes and encodes a given text prompt using a BPE (Byte Pair Encoding) tokenizer.

    The tokenizer configuration is loaded from a predefined JSON file located in the data directory.
    This function is typically used to prepare text inputs for processing by machine learning models.

    Args:
        prompt (str): The text prompt to be tokenized and encoded.

    Returns:
        Encoding: The tokenized and encoded version of the input text.
    """
    # Get the absolute path to the directory containing the script
    script_dir = os.path.abspath(os.path.dirname(__file__))
    # Construct the absolute path to the JSON file
    json_file_path = os.path.join(script_dir, "../data/tokenizer-wiki.json")
    tokenizer = Tokenizer.from_file(json_file_path)
    return tokenizer.encode(prompt)


def vertexai_encode(prompt: str = "") -> str:
    """
    A placeholder function for encoding text prompts intended for Vertex AI models.

    Currently, this function simplifies the prompt by removing spaces, which is not a typical behavior
    for actual encoding required by NLP models but serves as a placeholder for specific preprocessing.

    Args:
        prompt (str): The text prompt to be encoded.

    Returns:
        str: The simplified version of the input text.
    """
    return prompt.replace(" ", "")
