from tokenizers import Tokenizer, trainers, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from datasets import load_dataset

# to be implamented....
# models_costs = {
#     "Mistral-7B-v0.1": {
#         "prompt": 0.00,
#         "completion": 0.00,
#     },
#     "Mistral-7B-Instruct-v0.1": {
#         "prompt": 0.00,
#         "completion": 0.00,
#     }}



def tokenize_prompt(prompt, train_dataset, test_dataset, val_dataset):
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    def iterator_wiki(train_dataset, test_dataset, val_dataset):
        for mydataset in [train_dataset, test_dataset, val_dataset]:
            for data in mydataset:
                if isinstance(data.get("complex_sentence", None), str):
                    yield data["complex_sentence"]

    trainer = trainers.BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    tokenizer.train_from_iterator(iterator_wiki(train_dataset, test_dataset, val_dataset), trainer=trainer)

    output = tokenizer.encode(prompt)
    return len(output.tokens)

# Example usage
train_dataset = load_dataset('wiki_split', split='train')
test_dataset = load_dataset('wiki_split', split='test')
val_dataset = load_dataset('wiki_split', split='validation')

prompt = "Hello, y'all! How are you 😁 ?"
num_tokens = tokenize_prompt(prompt, train_dataset, test_dataset, val_dataset)
print(f"Number of tokens in the prompt: {num_tokens}")



#Encoding(num_tokens=11, attributes=[ids, type_ids, tokens, offsets, attention_mask, special_tokens_mask, overflowing])
