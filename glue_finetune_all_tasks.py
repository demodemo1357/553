import sys
import os
import argparse
import dataclasses
import functools
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import csv
import zipfile
import json
from datetime import datetime
from datasets import load_dataset
from transformers import BertTokenizer, BertModel as HFBertModel
from torch import Tensor
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr

from finetune import (
    BertConfig,
    BertModel,
    BertForSequenceClassification,
    load_pretrained_into_custom_bert
)


# ============================================================
# Configuration
# ============================================================

# All GLUE Tasks Configuration
GLUE_TASKS = {
    "cola": {"name": "CoLA", "metric": "matthews_correlation", "is_pair": False, "is_regression": False, "num_labels": 2},
    "sst2": {"name": "SST-2", "metric": "accuracy", "is_pair": False, "is_regression": False, "num_labels": 2},
    "mrpc": {"name": "MRPC", "metric": "f1", "is_pair": True, "is_regression": False, "num_labels": 2},
    "stsb": {"name": "STS-B", "metric": "pearson_spearman", "is_pair": True, "is_regression": True, "num_labels": 1},
    "qqp": {"name": "QQP", "metric": "f1", "is_pair": True, "is_regression": False, "num_labels": 2},
    "mnli": {"name": "MNLI", "metric": "accuracy", "is_pair": True, "is_regression": False, "num_labels": 3},
    "qnli": {"name": "QNLI", "metric": "accuracy", "is_pair": True, "is_regression": False, "num_labels": 2},
    "rte": {"name": "RTE", "metric": "accuracy", "is_pair": True, "is_regression": False, "num_labels": 2},
    "wnli": {"name": "WNLI", "metric": "accuracy", "is_pair": True, "is_regression": False, "num_labels": 2},
}

# Training Configuration
TRAIN_CONFIG = {
    "batch_size": 16,
    "max_len": 128,
    "learning_rate": 2e-5,
    "num_epochs": 3,
    "weight_decay": 0.01
}

# File naming mapping
NAME_MAP = {
    "sst2": "SST-2", "stsb": "STS-B", "cola": "CoLA", "mrpc": "MRPC",
    "qqp": "QQP", "mnli": "MNLI", "qnli": "QNLI", "rte": "RTE", "wnli": "WNLI"
}


# ============================================================
# Model Classes
# ============================================================

class BertForSequenceClassificationPair(nn.Module):
    """BERT model for sentence pair classification"""
    def __init__(self, config: BertConfig, num_labels: int):
        super().__init__()
        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(self, input_ids: Tensor, token_type_ids: Tensor = None, pad_id=0):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        _, pooled_output = self.bert(input_ids, token_type_ids, pad_id=pad_id)
        return self.classifier(pooled_output)


class BertForSequenceRegressionPair(nn.Module):
    """BERT model for sentence pair regression (STS-B)"""
    def __init__(self, config: BertConfig):
        super().__init__()
        self.bert = BertModel(config)
        self.regressor = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids: Tensor, token_type_ids: Tensor = None, pad_id=0):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        _, pooled_output = self.bert(input_ids, token_type_ids, pad_id=pad_id)
        return self.regressor(pooled_output)


# ============================================================
# Data Processing Functions
# ============================================================

def collate_glue_single(batch, tokenizer, max_len, is_regression=False):
    """Collate function for single sentence tasks"""
    texts = [x["sentence"] if "sentence" in x else x["text"] for x in batch]
    label_dtype = torch.float if is_regression else torch.long
    labels = torch.tensor([x["label"] for x in batch], dtype=label_dtype)
    encoded = tokenizer(texts, padding="max_length", truncation=True, max_length=max_len, 
                       return_tensors="pt", add_special_tokens=True)
    return encoded["input_ids"], labels


def collate_glue_pair(batch, tokenizer, max_len, is_regression=False):
    """Collate function for sentence pair tasks"""
    if "sentence1" in batch[0] and "sentence2" in batch[0]:
        texts = [(x["sentence1"], x["sentence2"]) for x in batch]
    elif "premise" in batch[0] and "hypothesis" in batch[0]:
        texts = [(x["premise"], x["hypothesis"]) for x in batch]
    elif "question" in batch[0] and "sentence" in batch[0]:
        texts = [(x["question"], x["sentence"]) for x in batch]
    elif "question1" in batch[0] and "question2" in batch[0]:
        texts = [(x["question1"], x["question2"]) for x in batch]
    else:
        raise ValueError(f"Unknown format: {batch[0].keys()}")

    label_dtype = torch.float if is_regression else torch.long
    labels = torch.tensor([x["label"] for x in batch], dtype=label_dtype)
    encoded = tokenizer(texts, padding="max_length", truncation=True, max_length=max_len, 
                       return_tensors="pt", add_special_tokens=True)
    return encoded["input_ids"], encoded["token_type_ids"], labels


def collate_test(batch, tokenizer, max_len, is_pair=False):
    """Collate function for test set (no labels)"""
    if is_pair:
        if "sentence1" in batch[0] and "sentence2" in batch[0]:
            texts = [(x["sentence1"], x["sentence2"]) for x in batch]
        elif "premise" in batch[0] and "hypothesis" in batch[0]:
            texts = [(x["premise"], x["hypothesis"]) for x in batch]
        elif "question" in batch[0] and "sentence" in batch[0]:
            texts = [(x["question"], x["sentence"]) for x in batch]
        elif "question1" in batch[0] and "question2" in batch[0]:
            texts = [(x["question1"], x["question2"]) for x in batch]
        else:
            raise ValueError(f"Unknown format: {batch[0].keys()}")
        encoded = tokenizer(texts, padding="max_length", truncation=True, max_length=max_len, 
                           return_tensors="pt", add_special_tokens=True)
        return encoded["input_ids"], encoded["token_type_ids"]
    else:
        texts = [x["sentence"] if "sentence" in x else x["text"] for x in batch]
        encoded = tokenizer(texts, padding="max_length", truncation=True, max_length=max_len, 
                           return_tensors="pt", add_special_tokens=True)
        return encoded["input_ids"]


# ============================================================
# Evaluation Functions
# ============================================================

def evaluate_model(model, val_loader, task_config, device, pad_id):
    """Evaluate model on validation set"""
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    loss_fn = nn.MSELoss() if task_config['is_regression'] else nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in val_loader:
            if task_config['is_pair']:
                input_ids, token_type_ids, labels = batch
                input_ids, token_type_ids, labels = input_ids.to(device), token_type_ids.to(device), labels.to(device)
                outputs = model(input_ids, token_type_ids, pad_id=pad_id)
            else:
                input_ids, labels = batch
                input_ids, labels = input_ids.to(device), labels.to(device)
                outputs = model(input_ids, pad_id=pad_id)

            if task_config['is_regression']:
                outputs = outputs.squeeze(-1) if outputs.dim() > 1 else outputs
                loss = loss_fn(outputs, labels)
                preds = outputs.cpu().numpy()
            else:
                loss = loss_fn(outputs, labels)
                preds = outputs.argmax(dim=-1).cpu().numpy()

            total_loss += loss.item()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(val_loader)

    if task_config['is_regression']:
        pearson_corr, _ = pearsonr(all_labels, all_preds)
        spearman_corr, _ = spearmanr(all_labels, all_preds)
        return {
            'pearson': pearson_corr, 
            'spearman': spearman_corr, 
            'pearson_spearman': (pearson_corr + spearman_corr) / 2, 
            'loss': avg_loss
        }
    else:
        return {
            'accuracy': accuracy_score(all_labels, all_preds), 
            'f1': f1_score(all_labels, all_preds, average='weighted'), 
            'mcc': matthews_corrcoef(all_labels, all_preds), 
            'loss': avg_loss
        }


# ============================================================
# Training Functions
# ============================================================

def train_single_task(task_key, task_config, device, tokenizer, pad_id, pretrained_state):
    """Train a single GLUE task and return training history"""
    print(f"\n{'='*60}")
    print(f"Training: {task_config['name']} ({task_key})")
    print(f"{'='*60}")

    # Load dataset
    print(f"Loading {task_key} dataset...")
    dataset = load_dataset("glue", task_key)
    print(f"  Train: {len(dataset['train'])}, Validation: {len(dataset['validation'])}")

    # Create model
    config = BertConfig()
    if task_config['is_regression']:
        model = BertForSequenceRegressionPair(config).to(device)
    elif task_config['is_pair']:
        model = BertForSequenceClassificationPair(config, task_config['num_labels']).to(device)
    else:
        model = BertForSequenceClassification(config, task_config['num_labels']).to(device)

    # Load pretrained weights
    load_pretrained_into_custom_bert(model.bert, pretrained_state)

    # Create data loaders
    if task_config['is_pair']:
        collate_fn = functools.partial(collate_glue_pair, tokenizer=tokenizer, 
                                       max_len=TRAIN_CONFIG['max_len'], 
                                       is_regression=task_config['is_regression'])
    else:
        collate_fn = functools.partial(collate_glue_single, tokenizer=tokenizer, 
                                       max_len=TRAIN_CONFIG['max_len'], 
                                       is_regression=task_config['is_regression'])

    train_loader = torch.utils.data.DataLoader(
        dataset['train'], batch_size=TRAIN_CONFIG['batch_size'], 
        shuffle=True, collate_fn=collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        dataset['validation'], batch_size=TRAIN_CONFIG['batch_size'], 
        shuffle=False, collate_fn=collate_fn
    )

    # Training setup
    loss_fn = nn.MSELoss() if task_config['is_regression'] else nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=TRAIN_CONFIG['learning_rate'], 
                            weight_decay=TRAIN_CONFIG['weight_decay'])

    history = {'train_loss': [], 'val_loss': [], 'val_metric': [], 'metric_name': task_config['metric']}

    # Training loop
    for epoch in range(TRAIN_CONFIG['num_epochs']):
        model.train()
        train_loss = 0

        for batch in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{TRAIN_CONFIG['num_epochs']}"):
            if task_config['is_pair']:
                input_ids, token_type_ids, labels = batch
                input_ids, token_type_ids, labels = input_ids.to(device), token_type_ids.to(device), labels.to(device)
                outputs = model(input_ids, token_type_ids, pad_id=pad_id)
            else:
                input_ids, labels = batch
                input_ids, labels = input_ids.to(device), labels.to(device)
                outputs = model(input_ids, pad_id=pad_id)

            if task_config['is_regression']:
                outputs = outputs.squeeze(-1) if outputs.dim() > 1 else outputs

            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        val_metrics = evaluate_model(model, val_loader, task_config, device, pad_id)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_metrics['loss'])

        if task_config['is_regression']:
            history['val_metric'].append(val_metrics['pearson_spearman'])
            print(f"  Epoch {epoch+1}: Loss={avg_train_loss:.4f}, Pearson={val_metrics['pearson']:.4f}, Spearman={val_metrics['spearman']:.4f}")
        elif task_config['metric'] == 'matthews_correlation':
            history['val_metric'].append(val_metrics['mcc'])
            print(f"  Epoch {epoch+1}: Loss={avg_train_loss:.4f}, MCC={val_metrics['mcc']:.4f}")
        elif task_config['metric'] == 'f1':
            history['val_metric'].append(val_metrics['f1'])
            print(f"  Epoch {epoch+1}: Loss={avg_train_loss:.4f}, F1={val_metrics['f1']:.4f}")
        else:
            history['val_metric'].append(val_metrics['accuracy'])
            print(f"  Epoch {epoch+1}: Loss={avg_train_loss:.4f}, Acc={val_metrics['accuracy']:.4f}")

    # Save model
    model_path = f"bert_model_glue_{task_key}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"  Model saved: {model_path}")

    return model, history, dataset


def train_mnli_only(device, tokenizer, pad_id, pretrained_state, all_models, all_histories, all_datasets, all_results):
    """Train MNLI task only (handles matched/mismatched validation sets)"""
    print("="*60)
    print("TRAINING MNLI ONLY")
    print("="*60)

    task_key = "mnli"
    task_config = GLUE_TASKS[task_key]

    print(f"\n{'='*60}")
    print(f"Training: {task_config['name']} ({task_key})")
    print(f"{'='*60}")

    # Load MNLI dataset
    print(f"Loading {task_key} dataset...")
    mnli_dataset = load_dataset("glue", task_key)
    print(f"  Train: {len(mnli_dataset['train'])}")
    print(f"  Validation (matched): {len(mnli_dataset['validation_matched'])}")
    print(f"  Validation (mismatched): {len(mnli_dataset['validation_mismatched'])}")
    print(f"  Test (matched): {len(mnli_dataset['test_matched'])}")
    print(f"  Test (mismatched): {len(mnli_dataset['test_mismatched'])}")

    # Create model
    config = BertConfig()
    mnli_model = BertForSequenceClassificationPair(config, task_config['num_labels']).to(device)

    # Load pretrained weights
    load_pretrained_into_custom_bert(mnli_model.bert, pretrained_state)

    # Create data loaders
    collate_fn = functools.partial(collate_glue_pair, tokenizer=tokenizer, 
                                   max_len=TRAIN_CONFIG['max_len'], is_regression=False)
    train_loader = torch.utils.data.DataLoader(
        mnli_dataset['train'], batch_size=TRAIN_CONFIG['batch_size'], 
        shuffle=True, collate_fn=collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        mnli_dataset['validation_matched'], batch_size=TRAIN_CONFIG['batch_size'], 
        shuffle=False, collate_fn=collate_fn
    )

    # Training setup
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(mnli_model.parameters(), lr=TRAIN_CONFIG['learning_rate'], 
                            weight_decay=TRAIN_CONFIG['weight_decay'])

    mnli_history = {'train_loss': [], 'val_loss': [], 'val_metric': [], 'metric_name': 'accuracy'}

    # Training loop
    for epoch in range(TRAIN_CONFIG['num_epochs']):
        mnli_model.train()
        train_loss = 0

        for batch in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{TRAIN_CONFIG['num_epochs']}"):
            input_ids, token_type_ids, labels = batch
            input_ids, token_type_ids, labels = input_ids.to(device), token_type_ids.to(device), labels.to(device)
            outputs = mnli_model(input_ids, token_type_ids, pad_id=pad_id)

            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Evaluate on validation_matched
        val_metrics = evaluate_model(mnli_model, val_loader, task_config, device, pad_id)

        mnli_history['train_loss'].append(avg_train_loss)
        mnli_history['val_loss'].append(val_metrics['loss'])
        mnli_history['val_metric'].append(val_metrics['accuracy'])

        print(f"  Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Acc (matched)={val_metrics['accuracy']:.4f}")

    # Also evaluate on validation_mismatched
    val_loader_mm = torch.utils.data.DataLoader(
        mnli_dataset['validation_mismatched'], batch_size=TRAIN_CONFIG['batch_size'], 
        shuffle=False, collate_fn=collate_fn
    )
    val_metrics_mm = evaluate_model(mnli_model, val_loader_mm, task_config, device, pad_id)
    print(f"\n  Final Val Acc (mismatched): {val_metrics_mm['accuracy']:.4f}")

    # Save model
    model_path = "bert_model_glue_mnli.pth"
    torch.save(mnli_model.state_dict(), model_path)
    print(f"\n  ✓ Model saved: {model_path}")

    # Store results
    all_models['mnli'] = mnli_model
    all_histories['mnli'] = mnli_history
    all_datasets['mnli'] = mnli_dataset
    all_results['mnli'] = {
        'best_metric': max(mnli_history['val_metric']),
        'metric_name': 'accuracy',
        'final_loss': mnli_history['val_loss'][-1],
        'val_acc_matched': val_metrics['accuracy'],
        'val_acc_mismatched': val_metrics_mm['accuracy']
    }

    print(f"\n{'='*60}")
    print("MNLI TRAINING COMPLETE!")
    print(f"  Best Val Acc (matched): {max(mnli_history['val_metric']):.4f}")
    print(f"  Final Val Acc (mismatched): {val_metrics_mm['accuracy']:.4f}")
    print(f"{'='*60}")

    return mnli_model, mnli_history, mnli_dataset


def train_all_tasks(device, tokenizer, pad_id, pretrained_state):
    """Train all GLUE tasks"""
    all_results = {}
    all_histories = {}
    all_models = {}
    all_datasets = {}

    start_time = datetime.now()
    print(f"\nStarting training at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Training {len(GLUE_TASKS)} tasks...\n")

    for task_key, task_config in GLUE_TASKS.items():
        try:
            # MNLI needs special handling due to matched/mismatched splits
            if task_key == "mnli":
                model, history, dataset = train_mnli_only(
                    device, tokenizer, pad_id, pretrained_state,
                    all_models, all_histories, all_datasets, all_results
                )
            else:
                model, history, dataset = train_single_task(
                    task_key, task_config, device, tokenizer, pad_id, pretrained_state
                )
                all_models[task_key] = model
                all_histories[task_key] = history
                all_datasets[task_key] = dataset
                all_results[task_key] = {
                    'best_metric': max(history['val_metric']),
                    'metric_name': history['metric_name'],
                    'final_loss': history['val_loss'][-1]
                }
        except Exception as e:
            print(f"Error training {task_key}: {e}")
            import traceback
            traceback.print_exc()

    end_time = datetime.now()
    print(f"\n{'='*60}")
    print(f"Training completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time: {end_time - start_time}")
    print(f"{'='*60}")

    return all_models, all_histories, all_datasets, all_results


# ============================================================
# Prediction and Submission Functions
# ============================================================

def load_model_for_task(task_key, task_config, device):
    """Load a trained model from .pth file"""
    config = BertConfig()

    if task_config['is_regression']:
        model = BertForSequenceRegressionPair(config)
    elif task_config['is_pair']:
        model = BertForSequenceClassificationPair(config, task_config['num_labels'])
    else:
        model = BertForSequenceClassification(config, task_config['num_labels'])

    model_path = f"bert_model_glue_{task_key}.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        print(f"  ✓ Loaded model from {model_path}")
        return model
    else:
        print(f"  ✗ Model file not found: {model_path}")
        return None


def generate_predictions_from_file(task_key, task_config, device, tokenizer, pad_id):
    """Generate predictions by loading model from file"""
    submission_files = []

    # Load model
    model = load_model_for_task(task_key, task_config, device)
    if model is None:
        return []

    model.eval()

    if task_key == "mnli":
        # MNLI has matched and mismatched test sets
        for split_name, split_key in [("matched", "test_matched"), ("mismatched", "test_mismatched")]:
            test_data = load_dataset("glue", "mnli", split=split_key)
            test_collate = functools.partial(collate_test, tokenizer=tokenizer, 
                                            max_len=TRAIN_CONFIG['max_len'], is_pair=True)
            test_loader = torch.utils.data.DataLoader(
                test_data, batch_size=TRAIN_CONFIG['batch_size'], 
                shuffle=False, collate_fn=test_collate
            )

            all_preds = []
            all_indices = [x['idx'] for x in test_data]

            with torch.no_grad():
                for batch in tqdm.tqdm(test_loader, desc=f"MNLI-{split_name}"):
                    input_ids, token_type_ids = batch
                    input_ids, token_type_ids = input_ids.to(device), token_type_ids.to(device)
                    outputs = model(input_ids, token_type_ids, pad_id=pad_id)
                    preds = outputs.argmax(dim=-1).cpu().numpy()
                    all_preds.extend(preds)

            filename = f"MNLI-{'m' if split_name == 'matched' else 'mm'}.tsv"
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(['index', 'prediction'])
                for idx, pred in zip(all_indices, all_preds):
                    writer.writerow([idx, pred])
            submission_files.append(filename)
            print(f"    ✓ Created {filename} ({len(all_preds)} predictions)")
    else:
        # Load test dataset
        dataset = load_dataset("glue", task_key)
        if 'test' not in dataset:
            print(f"    ✗ No test split found for {task_key}")
            return []

        test_collate = functools.partial(collate_test, tokenizer=tokenizer, 
                                        max_len=TRAIN_CONFIG['max_len'], 
                                        is_pair=task_config['is_pair'])
        test_loader = torch.utils.data.DataLoader(
            dataset['test'], batch_size=TRAIN_CONFIG['batch_size'], 
            shuffle=False, collate_fn=test_collate
        )

        all_preds = []
        all_indices = [x['idx'] for x in dataset['test']]

        with torch.no_grad():
            for batch in tqdm.tqdm(test_loader, desc=task_config['name']):
                if task_config['is_pair']:
                    input_ids, token_type_ids = batch
                    input_ids, token_type_ids = input_ids.to(device), token_type_ids.to(device)
                    outputs = model(input_ids, token_type_ids, pad_id=pad_id)
                else:
                    input_ids = batch
                    input_ids = input_ids.to(device)
                    outputs = model(input_ids, pad_id=pad_id)

                if task_config['is_regression']:
                    preds = outputs.squeeze(-1).cpu().numpy()
                    preds = np.clip(preds, 0.0, 5.0)
                    preds = np.round(preds, decimals=1)
                else:
                    preds = outputs.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)

        filename = f"{NAME_MAP.get(task_key, task_config['name'])}.tsv"
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['index', 'prediction'])
            for idx, pred in zip(all_indices, all_preds):
                writer.writerow([idx, pred])
        submission_files.append(filename)
        print(f"    ✓ Created {filename} ({len(all_preds)} predictions)")

    return submission_files


def generate_all_predictions(device, tokenizer, pad_id):
    """Generate predictions for all tasks and create submission ZIP"""
    print("="*60)
    print("GENERATING PREDICTIONS FOR ALL TASKS")
    print("(Loading models from saved .pth files)")
    print("="*60)

    # Check which model files exist
    print("\nChecking saved model files...")
    available_tasks = []
    for task_key in GLUE_TASKS:
        model_path = f"bert_model_glue_{task_key}.pth"
        if os.path.exists(model_path):
            available_tasks.append(task_key)
            print(f"  ✓ {model_path}")
        else:
            print(f"  ✗ {model_path} (not found)")

    print(f"\nFound {len(available_tasks)}/{len(GLUE_TASKS)} trained models")

    # Generate predictions for all available tasks
    all_submission_files = []

    for task_key in available_tasks:
        print(f"\n[{available_tasks.index(task_key)+1}/{len(available_tasks)}] Processing {GLUE_TASKS[task_key]['name']}...")
        files = generate_predictions_from_file(task_key, GLUE_TASKS[task_key], device, tokenizer, pad_id)
        all_submission_files.extend(files)

    # Generate AX predictions using MNLI model
    if 'mnli' in available_tasks:
        print(f"\n[AX] Generating AX predictions (using MNLI model)...")
        mnli_model = load_model_for_task('mnli', GLUE_TASKS['mnli'], device)
        if mnli_model is not None:
            mnli_model.eval()
            ax_data = load_dataset("glue", "ax", split="test")
            test_collate = functools.partial(collate_test, tokenizer=tokenizer, 
                                            max_len=TRAIN_CONFIG['max_len'], is_pair=True)
            test_loader = torch.utils.data.DataLoader(
                ax_data, batch_size=TRAIN_CONFIG['batch_size'], 
                shuffle=False, collate_fn=test_collate
            )

            all_preds = []
            all_indices = [x['idx'] for x in ax_data]

            with torch.no_grad():
                for batch in tqdm.tqdm(test_loader, desc="AX"):
                    input_ids, token_type_ids = batch
                    input_ids, token_type_ids = input_ids.to(device), token_type_ids.to(device)
                    outputs = mnli_model(input_ids, token_type_ids, pad_id=pad_id)
                    preds = outputs.argmax(dim=-1).cpu().numpy()
                    all_preds.extend(preds)

            with open('AX.tsv', 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(['index', 'prediction'])
                for idx, pred in zip(all_indices, all_preds):
                    writer.writerow([idx, pred])
            all_submission_files.append('AX.tsv')
            print(f"    ✓ Created AX.tsv ({len(all_preds)} predictions)")
    else:
        print("\n⚠ MNLI model not found, skipping AX predictions")

    # Create final ZIP
    print(f"\n{'='*60}")
    print("CREATING FINAL SUBMISSION ZIP")
    print(f"{'='*60}")

    zip_file = "GLUE_submission.zip"
    with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in all_submission_files:
            if os.path.exists(f):
                zf.write(f, os.path.basename(f))
                print(f"  Added: {f}")

    print(f"\n{'='*60}")
    print(f"✓ Final submission ZIP created: {zip_file}")
    print(f"  Size: {os.path.getsize(zip_file) / 1024:.2f} KB")
    print(f"  Contains {len(all_submission_files)} files")

    # Check if all 11 required files are present
    required_files = ['CoLA.tsv', 'SST-2.tsv', 'MRPC.tsv', 'STS-B.tsv', 'QQP.tsv',
                      'MNLI-m.tsv', 'MNLI-mm.tsv', 'QNLI.tsv', 'RTE.tsv', 'WNLI.tsv', 'AX.tsv']
    missing = [f for f in required_files if f not in all_submission_files]
    if missing:
        print(f"\n⚠ Missing files for complete submission: {missing}")
    else:
        print(f"\n✓ All 11 required files are present!")

    print(f"{'='*60}")
    print("You can now submit GLUE_submission.zip to the GLUE leaderboard.")
    print(f"{'='*60}")

    return all_submission_files


# ============================================================
# Visualization Functions
# ============================================================

def plot_training_curves(all_histories):
    """Plot training curves for all tasks"""
    num_tasks = len(all_histories)
    if num_tasks == 0:
        print("No training history to plot.")
        return

    cols = 3
    rows = (num_tasks + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows))
    if num_tasks > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for idx, (task_key, history) in enumerate(all_histories.items()):
        ax = axes[idx]
        epochs = range(1, len(history['train_loss']) + 1)

        ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', marker='o')
        ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', marker='s')

        ax2 = ax.twinx()
        ax2.plot(epochs, history['val_metric'], 'g-', label=history['metric_name'], marker='^')
        ax2.set_ylabel(history['metric_name'], color='g')
        ax2.tick_params(axis='y', labelcolor='g')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f"{GLUE_TASKS[task_key]['name']}")
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for idx in range(len(all_histories), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig('glue_training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Training curves saved to: glue_training_curves.png")


def print_results_summary(all_results):
    """Print final results summary"""
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    print(f"{'Task':<10} {'Metric':<25} {'Best Score':<15}")
    print("-"*60)

    for task_key, result in all_results.items():
        print(f"{GLUE_TASKS[task_key]['name']:<10} {result['metric_name']:<25} {result['best_metric']:.4f}")

    print("="*60)

    # Save results to JSON
    with open('glue_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("Results saved to: glue_results.json")


# ============================================================
# Main Function
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='GLUE Benchmark Training Pipeline')
    parser.add_argument('--train', action='store_true', help='Train all GLUE tasks')
    parser.add_argument('--predict', action='store_true', help='Generate predictions and create submission ZIP')
    parser.add_argument('--plot', action='store_true', help='Plot training curves')
    args = parser.parse_args()

    # If no arguments provided, run full pipeline
    run_all = not (args.train or args.predict or args.plot)

    # Setup
    print("="*60)
    print("GLUE Benchmark - Automated Training Pipeline")
    print("="*60)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"Using Device: {device}")
    print(f"Tasks to train: {list(GLUE_TASKS.keys())}")
    print(f"Total tasks: {len(GLUE_TASKS)}")
    print("="*60)

    # Load tokenizer and pretrained weights
    print("\nLoading tokenizer and pretrained BERT weights...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    pad_id = tokenizer.pad_token_id
    hf_bert = HFBertModel.from_pretrained("bert-base-uncased")
    pretrained_state = hf_bert.state_dict()
    print("Done!")

    all_models = {}
    all_histories = {}
    all_datasets = {}
    all_results = {}

    # Training
    if args.train or run_all:
        all_models, all_histories, all_datasets, all_results = train_all_tasks(
            device, tokenizer, pad_id, pretrained_state
        )

        # Print results summary
        if all_results:
            print_results_summary(all_results)

    # Plot training curves
    if args.plot or run_all:
        if all_histories:
            plot_training_curves(all_histories)
        else:
            # Try to load from JSON if available
            if os.path.exists('glue_results.json'):
                with open('glue_results.json', 'r') as f:
                    all_results = json.load(f)
                print_results_summary(all_results)
            else:
                print("No training history available for plotting.")

    # Generate predictions
    if args.predict or run_all:
        generate_all_predictions(device, tokenizer, pad_id)


if __name__ == "__main__":
    main()

