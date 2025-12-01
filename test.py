import torch
import torch.nn as nn
from transformers import BertTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
import tqdm

# -------------------------------
# Import your custom BERT classes
# -------------------------------
from finetune import BertConfig, BertModel, BertForSequenceClassification


# -------------------------------
# Load tokenizer
# -------------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
pad_id = tokenizer.pad_token_id


# -------------------------------
# Load dataset
# -------------------------------
dataset = load_dataset("tum-nlp/neural-news-benchmark")

# same normalization used in train
def normalize_label(example):
    if example["label"] == "real":
        return {"label": "human"}
    return example

dataset = dataset.map(normalize_label)
dataset["test"] = dataset["test"].filter(lambda x: x["language"] == "en")

label_to_id = {"human": 0, "neural": 1}


# -------------------------------
# Collate function (same as train)
# -------------------------------
def collate_fn(batch):
    texts = [x["body"] for x in batch]
    labels = torch.tensor([label_to_id[x["label"]] for x in batch])

    input_ids = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )["input_ids"]

    return input_ids, labels


# -------------------------------
# Load your fine-tuned model
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = BertConfig()
model = BertForSequenceClassification(config, num_labels=2)
model.load_state_dict(torch.load("bert_model_neural_news.pth", map_location=device))
model.to(device)
model.eval()

print("Loaded fine-tuned model!")


# -------------------------------
# Evaluate on test set
# -------------------------------
test_loader = DataLoader(dataset["test"], batch_size=16, shuffle=False, collate_fn=collate_fn)

correct = 0
total = 0

with torch.no_grad():
    for input_ids, labels in tqdm.tqdm(test_loader, desc="Testing"):
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        logits = model(input_ids, pad_id=pad_id)
        pred = logits.argmax(dim=-1)

        correct += (pred == labels).sum().item()
        total += len(labels)

acc = correct / total
print(f"\nFinal Test Accuracy: {acc:.4f}")


# -------------------------------
# Optional: single-sentence testing
# -------------------------------
def classify_text(text):
    input_ids = tokenizer(
        [text],
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )["input_ids"].to(device)

    logits = model(input_ids, pad_id=pad_id)
    pred = logits.argmax(dim=-1).item()
    return "human" if pred == 0 else "neural"


# Example usage
print("\n--- Single Sentence Test ---")
sample = "This is a breaking news report from Reuters."
print(sample, "->", classify_text(sample))

sample = "The research demonstrates how quantum data transformers optimize meta-scaling efficiency."
print(sample, "->", classify_text(sample))
