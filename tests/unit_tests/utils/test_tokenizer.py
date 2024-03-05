import unittest

from proxyllm.utils.tokenizer import bpe_tokenize_encode


def test_invalid_character_input():
    # Arrange
    input = "Hello, y'all! How are you 😁 ?"

    # Act
    res = bpe_tokenize_encode(input)

    # Assert
    assert "[UNK]" in res.tokens


def test_valid_character_input():
    # Arrange
    input = "Hello, y'all! How are you doing today?"

    # Act
    res = bpe_tokenize_encode(input)

    # Assert
    assert len(res.tokens) == 12


def test_short_input():
    # Arrange
    input = "yo,"

    # Act
    res = bpe_tokenize_encode(input)

    # Assert
    assert len(res.tokens) == 2


def test_long_input():
    # Arrange
    input = "The quick brown fox jumps over the lazy dog, leaping through the tall grass, chasing its tail in circles, and barking loudly, as the birds in the nearby trees sing their melodious songs, creating a cacophony of nature's sounds that echo through the forest, bringing peace and serenity to all who listen."

    # Act
    res = bpe_tokenize_encode(input)

    # Assert
    assert len(res.tokens) == 73


def test_empty_string_input():
    # Arrange
    input = ""

    # Act
    res = bpe_tokenize_encode(input)

    # Assert
    assert len(res.tokens) == 0


def test_all_capital_input():
    # Arrange
    input = (
        "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG AND LANDS SAFELY ON THE OTHER SIDE"
    )

    # Act
    res = bpe_tokenize_encode(input)

    for token in res.tokens:
        assert token.isupper()


def test_all_space_input():
    # Arrange
    space = " "
    input = space * 9
    # Act
    res = bpe_tokenize_encode(input)

    assert len(res.tokens) == 0


class TestBpeTokenizationTypeError(unittest.TestCase):
    def test_bpe_tokenize_encode_type_error_int(self):
        # AAA
        with self.assertRaises(TypeError):
            bpe_tokenize_encode(25)

    def test_bpe_tokenize_encode_type_error_float(self):
        # AAA
        with self.assertRaises(TypeError):
            bpe_tokenize_encode(1.01)

    def test_bpe_tokenize_encode_type_error_bool(self):
        # AAA
        with self.assertRaises(TypeError):
            bpe_tokenize_encode(True)

    def test_bpe_tokenize_encode_type_error_None(self):
        # AAA
        with self.assertRaises(TypeError):
            bpe_tokenize_encode(True)


if __name__ == "__main__":
    unittest.main()
