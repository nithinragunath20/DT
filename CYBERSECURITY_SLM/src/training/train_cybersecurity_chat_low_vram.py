import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast
from model import Qwen3Model


def get_state(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "model_state", "model", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise ValueError("Invalid checkpoint format")
    if all(k.startswith("module.") for k in checkpoint):
        checkpoint = {k[7:]: v for k, v in checkpoint.items()}
    return checkpoint


def prepare_data(records, tokenizer, vocab_size, max_tokens):
    samples = []
    for row in records:
        question = row["question"].strip()
        answer = row["answer"].strip()
        prefix = f"Question: {question}\nAnswer: "

        p = tokenizer.encode(prefix, add_special_tokens=False)
        a = tokenizer.encode(answer, add_special_tokens=False)
        ids = p + a

        if not p or not a or len(ids) > max_tokens:
            continue
        if max(ids) >= vocab_size:
            continue

        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor([-100] * (len(p) - 1) + a,
                         dtype=torch.long)
        samples.append((x, y))
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
    parser.add_argument("--max-tokens", type=int, default=96)
    args = parser.parse_args()

    base = Path(r"D:\SLM")
    cfg = json.loads((base / "export/config.json").read_text())

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(base / "export/tokenizer.json")
    )

    with open(base / "instruction_data/cybersecurity_qa.jsonl",
              encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    random.seed(42)
    random.shuffle(records)

    split = max(1, len(records) // 10)
    train = prepare_data(
        records[split:], tokenizer, cfg["vocab_size"],
        min(args.max_tokens, cfg["max_seq_len"])
    )
    valid = prepare_data(
        records[:split], tokenizer, cfg["vocab_size"],
        min(args.max_tokens, cfg["max_seq_len"])
    )

    if not train or not valid:
        raise ValueError("No valid training or validation samples")

    print(f"Train: {len(train)} | Validation: {len(valid)}")

    device = args.device or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    keys = [
        "vocab_size", "d_model", "n_heads",
        "n_kv_groups", "n_layers", "d_ff", "max_seq_len"
    ]
    model = Qwen3Model(**{k: cfg[k] for k in keys})

    checkpoint = torch.load(
        base / "checkpoints/best_model.pt",
        map_location="cpu",
        weights_only=True
    )
    model.load_state_dict(get_state(checkpoint), strict=True)

    # Freeze large layers to save GPU memory.
    model.tok_emb.requires_grad_(False)
    model.out_head.requires_grad_(False)

    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=2e-5)

    use_amp = device == "cuda"
    amp_dtype = (
        torch.bfloat16
        if use_amp and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=use_amp and amp_dtype == torch.float16
    )

    def calculate_loss(x, y):
        with torch.autocast(
            device_type="cuda", dtype=amp_dtype, enabled=use_amp
        ):
            logits, _ = model(x.unsqueeze(0).to(device))
            return F.cross_entropy(
                logits.float().reshape(-1, cfg["vocab_size"]),
                y.to(device).reshape(-1),
                ignore_index=-100
            )

    output = base / "chat_finetuned"
    output.mkdir(exist_ok=True)
    best_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        random.shuffle(train)
        total = 0.0

        for i, (x, y) in enumerate(train, 1):
            optimizer.zero_grad(set_to_none=True)
            loss = calculate_loss(x, y)

            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer)
            scaler.update()

            total += loss.item()
            if i % 5 == 0:
                print(f"Epoch {epoch+1} Step {i}/{len(train)} Loss {loss.item():.4f}")

        model.eval()
        with torch.no_grad():
            val_loss = sum(
                calculate_loss(x, y).item() for x, y in valid
            ) / len(valid)

        print(f"Epoch {epoch+1}: Train={total/len(train):.4f} Validation={val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                model.state_dict(),
                output / "best_chat_model.pt"
            )
            print("Checkpoint saved successfully!")

    print("Training pipeline test completed.")


if __name__ == "__main__":
    main()
