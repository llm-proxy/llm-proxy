from tokenizers import Tokenizer

from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

tokenizer = Tokenizer(BPE())

trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer.pre_tokenizer = Whitespace()
tokenizer.train(
    files=["wiki.train.raw", "wiki.valid.raw", "wiki.test.raw"], trainer=trainer
)
tokenizer.save("llmproxy/data/tokenizer-wiki-new.json")
tokenizer = Tokenizer.from_file("llmproxy/data/tokenizer-wiki-new.json")
output = tokenizer.encode("Hello, y'all! How are you 😁 ?")

print(output.tokens)
