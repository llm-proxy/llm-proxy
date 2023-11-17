from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import evaluate

dataset = load_dataset('csv', data_files={'train': 'train.csv', 'test': 'test.csv'})
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

# Preprocess the dataset to be model readable
def tokenization(batch):
    return tokenizer(batch['text'], truncation=True, padding=True)

metric = evaluate.load("accuracy")

# To measure accuracy of model
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

encoded_dataset = dataset.map(tokenization, batched=True)
classification_model = AutoModelForSequenceClassification.from_pretrained('distilbert-based-uncased', num_labels=10)
training_args = TrainingArguments('test_trainer', evaluation_strategy="epoch")

trainer = Trainer(
    model=classification_model, 
    args=training_args, 
    train_dataset=encoded_dataset["train"], 
    eval_dataset=encoded_dataset["test"],
    compute_metrics=compute_metrics
)

trainer.train()
classification_model.save_pretrained("llmproxy/data/classification")