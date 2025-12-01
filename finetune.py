import dataclasses
import functools

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from datasets import load_dataset
from transformers import BertTokenizer, BertModel as HFBertModel
from torch import Tensor


# BERT config and model defined previously
@dataclasses.dataclass
class BertConfig:
    vocab_size: int = 30522
    num_layers: int = 12
    hidden_size: int = 768
    num_heads: int = 12
    dropout_prob: float = 0.1
    pad_id: int = 0
    max_seq_len: int = 512
    num_types: int = 2


class BertBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout_prob: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout_prob, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.ff_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )

    def forward(self, x: Tensor, pad_mask: Tensor) -> Tensor:
        attn_output, _ = self.attention(
            x, x, x, key_padding_mask=pad_mask
        )
        x = self.attn_norm(x + attn_output)
        ff_output = self.feed_forward(x)
        x = self.ff_norm(x + self.dropout(ff_output))
        return x


class BertPooler(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.dense(x))


class BertModel(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_id
        )
        self.type_embeddings = nn.Embedding(config.num_types, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_seq_len, config.hidden_size)
        self.embeddings_norm = nn.LayerNorm(config.hidden_size)
        self.embeddings_dropout = nn.Dropout(config.dropout_prob)

        self.blocks = nn.ModuleList(
            [
                BertBlock(config.hidden_size, config.num_heads, config.dropout_prob)
                for _ in range(config.num_layers)
            ]
        )
        self.pooler = BertPooler(config.hidden_size)

    def forward(self, input_ids: Tensor, token_type_ids: Tensor, pad_id=0):
        pad_mask = input_ids == pad_id

        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        token_embeddings = self.word_embeddings(input_ids)
        type_embeddings = self.type_embeddings(token_type_ids)
        pos_embeddings = self.position_embeddings(position_ids)

        x = token_embeddings + type_embeddings + pos_embeddings
        x = self.embeddings_dropout(self.embeddings_norm(x))

        for block in self.blocks:
            x = block(x, pad_mask)

        pooled_output = self.pooler(x[:, 0])
        return x, pooled_output


class BertForSequenceClassification(nn.Module):
    def __init__(self, config: BertConfig, num_labels: int):
        super().__init__()
        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(self, input_ids: Tensor, pad_id=0):
        token_type_ids = torch.zeros_like(input_ids)
        _, pooled_output = self.bert(input_ids, token_type_ids, pad_id=pad_id)
        return self.classifier(pooled_output)


# -------------------------------
# Load neural-news dataset
# -------------------------------
dataset = load_dataset("tum-nlp/neural-news-benchmark")

# 把 'real' 归一化成 'human'，做二分类：human vs neural
def normalize_label(example):
    label = example["label"]
    if label == "real":
        label = "human"
    # 如果以后还有奇怪的标签，可以在这里继续处理
    return {"label": label}

dataset = dataset.map(normalize_label)

dataset["train"] = dataset["train"].filter(lambda x: x["language"] == "en")
dataset["validation"] = dataset["validation"].filter(lambda x: x["language"] == "en")

label_to_id = {"human": 0, "neural": 1}
num_labels = 2


# -------------------------------
# HuggingFace tokenizer
# -------------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def collate_neural_news(batch, tokenizer, max_len):
    texts = [x["body"] for x in batch]
    labels = torch.tensor([label_to_id[x["label"]] for x in batch])

    input_ids = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
        add_special_tokens=True,
    )["input_ids"]

    return input_ids, labels


batch_size = 16
max_len = 128
collate_fn = functools.partial(collate_neural_news, tokenizer=tokenizer, max_len=max_len)

train_loader = torch.utils.data.DataLoader(
    dataset["train"],
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
)
val_loader = torch.utils.data.DataLoader(
    dataset["validation"],
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
)


# -------------------------------
# Load HF pretrained weights into custom BERT
# -------------------------------
def load_pretrained_into_custom_bert(custom_bert, hf_state):
    custom_bert.word_embeddings.weight.data.copy_(hf_state["embeddings.word_embeddings.weight"])
    custom_bert.position_embeddings.weight.data.copy_(hf_state["embeddings.position_embeddings.weight"])
    custom_bert.type_embeddings.weight.data.copy_(hf_state["embeddings.token_type_embeddings.weight"])
    custom_bert.embeddings_norm.weight.data.copy_(hf_state["embeddings.LayerNorm.weight"])
    custom_bert.embeddings_norm.bias.data.copy_(hf_state["embeddings.LayerNorm.bias"])

    for i, block in enumerate(custom_bert.blocks):
        prefix = f"encoder.layer.{i}."

        qw = hf_state[prefix + "attention.self.query.weight"]
        kw = hf_state[prefix + "attention.self.key.weight"]
        vw = hf_state[prefix + "attention.self.value.weight"]

        qb = hf_state[prefix + "attention.self.query.bias"]
        kb = hf_state[prefix + "attention.self.key.bias"]
        vb = hf_state[prefix + "attention.self.value.bias"]

        block.attention.in_proj_weight.data.copy_(torch.cat([qw, kw, vw], dim=0))
        block.attention.in_proj_bias.data.copy_(torch.cat([qb, kb, vb], dim=0))

        block.attention.out_proj.weight.data.copy_(hf_state[prefix + "attention.output.dense.weight"])
        block.attention.out_proj.bias.data.copy_(hf_state[prefix + "attention.output.dense.bias"])

        block.attn_norm.weight.data.copy_(hf_state[prefix + "attention.output.LayerNorm.weight"])
        block.attn_norm.bias.data.copy_(hf_state[prefix + "attention.output.LayerNorm.bias"])

        block.feed_forward[0].weight.data.copy_(hf_state[prefix + "intermediate.dense.weight"])
        block.feed_forward[0].bias.data.copy_(hf_state[prefix + "intermediate.dense.bias"])

        block.feed_forward[2].weight.data.copy_(hf_state[prefix + "output.dense.weight"])
        block.feed_forward[2].bias.data.copy_(hf_state[prefix + "output.dense.bias"])

        block.ff_norm.weight.data.copy_(hf_state[prefix + "output.LayerNorm.weight"])
        block.ff_norm.bias.data.copy_(hf_state[prefix + "output.LayerNorm.bias"])

    custom_bert.pooler.dense.weight.data.copy_(hf_state["pooler.dense.weight"])
    custom_bert.pooler.dense.bias.data.copy_(hf_state["pooler.dense.bias"])

    print("Loaded HuggingFace BERT weights!")


# -------------------------------
# Init model + load pretrained
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = BertConfig()
model = BertForSequenceClassification(config, num_labels).to(device)

hf_bert = HFBertModel.from_pretrained("bert-base-uncased")
hf_state = hf_bert.state_dict()
load_pretrained_into_custom_bert(model.bert, hf_state)


# -------------------------------
# Train
# -------------------------------
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
pad_id = tokenizer.pad_token_id

for epoch in range(3):
    model.train()
    for input_ids, labels in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/3"):
        input_ids, labels = input_ids.to(device), labels.to(device)

        logits = model(input_ids, pad_id=pad_id)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    total, correct, val_loss = 0, 0, 0
    with torch.no_grad():
        for input_ids, labels in val_loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            logits = model(input_ids, pad_id=pad_id)

            loss = loss_fn(logits, labels)
            val_loss += loss.item()

            pred = logits.argmax(dim=-1)
            correct += (pred == labels).sum().item()
            total += len(labels)

    print(f"Val acc: {correct/total:.4f}, loss: {val_loss/len(val_loader):.4f}")


torch.save(model.state_dict(), "bert_model_neural_news.pth")
