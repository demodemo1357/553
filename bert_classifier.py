import argparse
import os
import pandas as pd
import numpy as np
from datasets import Dataset
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments
)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

class BertClassifier:
    def __init__(self, model_name="distilbert/distilbert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.id2label = {0: "real", 1: "fake", 2: "neural"}
        self.label2id = {"real": 0, "fake": 1, "neural": 2}
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3,
            id2label=self.id2label,
            label2id=self.label2id
        )

    def load_data(self, csv_path):
        print("Loading CSV...")
        df = pd.read_csv(csv_path)
        print("Reassigning labels...")
        mapping = {
            "true": 0,
            "fake": 1,
            "neural": 2
        }
        df["label"] = df["source"].map(mapping)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        print("Splitting data...")
        train_val, test = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df["label"]
        )
        train, valid = train_test_split(
            train_val, test_size=0.125, random_state=42, stratify=train_val["label"]
        )
        return train, valid, test

    def prepare_datasets(self, train, valid, test):
        train_ds = Dataset.from_pandas(train)
        valid_ds = Dataset.from_pandas(valid)
        test_ds = Dataset.from_pandas(test)
        def preprocess(examples):
            return self.tokenizer(examples["text"], truncation=True, padding=True)
        print("Tokenizing datasets...")
        tokenized_train = train_ds.map(preprocess, batched=True)
        tokenized_valid = valid_ds.map(preprocess, batched=True)
        tokenized_test = test_ds.map(preprocess, batched=True)
        data_collator = DataCollatorWithPadding(self.tokenizer)
        return tokenized_train, tokenized_valid, tokenized_test, data_collator

    def train(self, tokenized_train, tokenized_valid, data_collator, output_dir):
        print("Configuring training...")
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=2,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            push_to_hub=False,
            report_to="none",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_valid,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        print("Training BERT model...")
        trainer.train()
        print("Training finished!")
        return trainer


    def evaluate(self, trainer, tokenized_test):
        print("Evaluating model on test set...")
        results = trainer.predict(tokenized_test)

        preds = np.argmax(results.predictions, axis=1)
        labels = results.label_ids

        print("\n--- Test Set Performance ---")
        print(f"Accuracy:  {results.metrics['test_accuracy']:.4f}")
        print(f"Precision: {results.metrics['test_precision']:.4f}")
        print(f"Recall:    {results.metrics['test_recall']:.4f}")
        print(f"F1 Score:  {results.metrics['test_f1']:.4f}")

        print("\nDetailed Classification Report:")
        report = classification_report(labels, preds, target_names=['real', 'fake', 'neural'])
        print(report)

def main():
    parser = argparse.ArgumentParser(description="Train a BERT classifier on benchmark.csv")
    parser.add_argument("--csv", type=str, required=True, help="Path to benchmark.csv")
    parser.add_argument("--output_dir", type=str, default="my_awesome_model", help="Model output directory")
    args = parser.parse_args()

    clf = BertClassifier()
    train, valid, test = clf.load_data(args.csv)
    tokenized_train, tokenized_valid, tokenized_test, collator = clf.prepare_datasets(train, valid, test)
    trainer = clf.train(tokenized_train, tokenized_valid, collator, args.output_dir)
    clf.evaluate(trainer, tokenized_test)


if __name__ == "__main__":
    main()

