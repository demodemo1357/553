import tokenizers
from datasets import load_dataset

# Use WikiText-2 instead of WikiText-103
path, name = "wikitext", "wikitext-2-raw-v1"
vocab_size = 30522

# Load dataset
dataset = load_dataset(path, name, split="train")

# Collect texts, skip title lines starting with "=" (same as your style)
texts = []
for line in dataset["text"]:
    line = line.strip()
    if line and not line.startswith("="):
        texts.append(line)

# Configure WordPiece tokenizer
tokenizer = tokenizers.Tokenizer(tokenizers.models.WordPiece())
tokenizer.normalizer = tokenizers.normalizers.NFKC()
tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
tokenizer.decoder = tokenizers.decoders.WordPiece()

trainer = tokenizers.trainers.WordPieceTrainer(
    vocab_size=vocab_size,
    special_tokens=["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"]
)

# Train
tokenizer.train_from_iterator(texts, trainer=trainer)

# Optional: enable padding
tokenizer.enable_padding(
    pad_id=tokenizer.token_to_id("[PAD]"),
    pad_token="[PAD]"
)

# Save tokenizer
tokenizer_path = "wikitext-2_wordpiece.json"
tokenizer.save(tokenizer_path, pretty=True)

# Test
tokenizer = tokenizers.Tokenizer.from_file(tokenizer_path)
encoding = tokenizer.encode("Hello, world!")
print(encoding.tokens)
print(tokenizer.decode(encoding.ids))
