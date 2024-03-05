from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

pre_tokenizer = pre_tokenizers.Metaspace(
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
)
tokenizer.pre_tokenizer = Whitespace()
tokenizer.train(
    files=["wiki.train.raw", "wiki.valid.raw", "wiki.test.raw"], trainer=trainer
)
tokenizer.save("llmproxy/data/tokenizer-wiki-new.json")
tokenizer = Tokenizer.from_file("llmproxy/data/tokenizer-wiki-new.json")
output = tokenizer.encode("Hello, y'all! How are you  ?")
