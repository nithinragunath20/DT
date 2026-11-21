"""Continue instruction training of the ORIGINAL custom Cybersecurity SLM.

Place beside model.py and config.json in D:\\SLM (or run with --project).
Requires: pip install torch transformers tokenizers
Dataset: JSONL, each row {"question": "...", "answer": "..."}.

This does NOT create a pretrained replacement or guarantee factual accuracy.
"""
import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerFast

from model import Qwen3Model


def checkpoint_state(obj):
    if isinstance(obj, dict):
        for key in ("model_state_dict", "model_state", "model", "state_dict"):
            candidate = obj.get(key)
            if isinstance(candidate, dict) and any(torch.is_tensor(v) for v in candidate.values()):
                obj = candidate
                break
    if not isinstance(obj, dict):
        raise ValueError("Checkpoint does not contain a model state dict")
    if all(k.startswith("module.") for k in obj):
        obj = {k[7:]: v for k, v in obj.items()}
    return obj


class QADataset(Dataset):
    def __init__(self, records, tokenizer, vocab_size, context):
        self.samples = []
        self.skipped = 0
        for row in records:
            question, answer = row["question"].strip(), row["answer"].strip()
            if not question or not answer:
                self.skipped += 1
                continue
            # Separate tokenization avoids added special tokens outside the original vocab.
            prefix = f"Question: {question}\nAnswer: "
            p_ids = tokenizer.encode(prefix, add_special_tokens=False)
            a_ids = tokenizer.encode(answer, add_special_tokens=False)
            ids = p_ids + a_ids
            if len(ids) < 2 or len(ids) > context or not a_ids:
                self.skipped += 1
                continue
            if min(ids) < 0 or max(ids) >= vocab_size:
                self.skipped += 1
                continue
            # Input predicts next token. Only answer tokens contribute to loss.
            x = ids[:-1]
            y = [-100] * (len(p_ids) - 1) + a_ids
            assert len(x) == len(y)
            self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def make_collate(pad_id):
    def collate(batch):
        max_len = max(len(x) for x, _ in batch)
        xs, ys = [], []
        for x, y in batch:
            xs.append(x + [pad_id] * (max_len - len(x)))
            ys.append(y + [-100] * (max_len - len(y)))
        return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)
    return collate


def loss_for_batch(model, xb, yb):
    logits, _ = model(xb)
    # Original model.py's built-in loss does not ignore masked labels.
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                           yb.reshape(-1), ignore_index=-100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path(r"D:\SLM"))
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project = args.project
    config_path = project / "export" / "config.json"
    tokenizer_path = project / "export" / "tokenizer.json"
    checkpoint_path = args.checkpoint or project / "checkpoints" / "best_model.pt"
    data_path = args.data or project / "instruction_data" / "cybersecurity_qa.jsonl"
    output_dir = project / "chat_finetuned"
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in (config_path, tokenizer_path, checkpoint_path, data_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path))
    # The original tokenizer has 26 added-token IDs above model's vocabulary.
    # This script excludes examples containing any out-of-range tokens.
    vocab_size = cfg["vocab_size"]
    pad_id = 0  # Only for padding. Padding targets are ignored.
    if pad_id >= vocab_size:
        raise ValueError("Invalid padding token")

    with data_path.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if len(records) < 10:
        raise ValueError("Provide at least 10 Q&A examples for a smoke test; hundreds/thousands of reviewed examples are preferable.")
    random.shuffle(records)
    n_valid = max(1, round(len(records) * .1))
    val_records, train_records = records[:n_valid], records[n_valid:]
    train_ds = QADataset(train_records, tokenizer, vocab_size, cfg["max_seq_len"])
    val_ds = QADataset(val_records, tokenizer, vocab_size, cfg["max_seq_len"])
    if not train_ds or not val_ds:
        raise ValueError("No valid samples in one split. Shorten answers or check tokenizer IDs.")
    print(f"Train: {len(train_ds)} | Validation: {len(val_ds)} | "
          f"Skipped: {train_ds.skipped + val_ds.skipped}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = Qwen3Model(**cfg_no_dtype(cfg))
    try:
        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError("Could not safely load checkpoint. Inspect its format; do not disable weights_only for untrusted files.") from exc
    model.load_state_dict(checkpoint_state(loaded), strict=True)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    collate = make_collate(pad_id)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate)

    best_val = math.inf
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for step, (xb, yb) in enumerate(train_loader, start=1):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for_batch(model, xb, yb)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss in epoch {epoch}, batch {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
            if step % 20 == 0:
                print(f"epoch {epoch} batch {step}/{len(train_loader)} loss={loss.item():.4f}")
        model.eval()
        valid_total = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                valid_total += loss_for_batch(model, xb, yb).item()
        val_loss = valid_total / len(val_loader)
        print(f"Epoch {epoch}: train={total / len(train_loader):.4f} val={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            output_path = output_dir / "best_chat_model.pt"
            torch.save(model.cpu().state_dict(), output_path)
            model.to(device)
            print(f"Saved improved checkpoint: {output_path}")
    print("Training finished. Evaluate factual accuracy and abstention on held-out questions before GGUF conversion.")


def cfg_no_dtype(cfg):
    return {k: cfg[k] for k in ("vocab_size", "d_model", "n_heads", "n_kv_groups",
                                 "n_layers", "d_ff", "max_seq_len")}


if __name__ == "__main__":
    main()
