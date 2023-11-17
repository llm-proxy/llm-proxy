from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import numpy as np
import evaluate

print("loading dataset")
dataset = load_dataset("csv", data_files={"train": "train.csv", "test": "test.csv"})
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
print("dataset loaded")

# Preprocess the dataset to be model readable
def tokenization(batch):
    inputs = tokenizer(batch["prompt"], truncation=True, padding=True)
    inputs["labels"] = batch["category"]
    return inputs

metric = evaluate.load("accuracy")

# To measure accuracy of model
def compute_metrics(p):
    logits, labels = p.predictions, p.label_ids
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

encoded_dataset = dataset.map(tokenization, batched=True)
classification_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=10)
training_args = TrainingArguments("test_trainer", evaluation_strategy="epoch")

trainer = Trainer(
    model=classification_model, 
    args=training_args, 
    train_dataset=encoded_dataset["train"], 
    eval_dataset=encoded_dataset["test"],
    compute_metrics=compute_metrics,
)

# Custom training step
def training_step(model, inputs):
    # Print the keys of the first training example
    if isinstance(inputs, dict):
        print("Keys of the first training example:", inputs.keys())
    
    labels = inputs.pop("category", None)  # Try to get "category" key, default to None if not present
    if labels is not None:
        inputs["labels"] = labels
    outputs = model(**inputs)
    
    # Check if the loss is not None
    if outputs.loss is not None:
        loss = outputs.loss
    else:
        loss = torch.tensor(0.0, device=outputs.logits.device)
    
    return loss

trainer.training_step = training_step

print("TRAINING MODEL...")
trainer.train()
print("TRAINING DONE!")
print("SAVING MODEL...")
classification_model.save_pretrained("llmproxy/data/classification")
print("MODEL SAVED!")
