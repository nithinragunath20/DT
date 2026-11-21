# Cybersecurity SLM — Custom Transformer & Ollama Deployment

A research prototype that trains a **custom decoder-only Transformer language model** on DARPA Transparent Computing **CDM18-style telemetry**, converts its trained weights to GGUF, and runs it locally with Ollama. A **separate** Qwen3 0.6B workflow can explain generated telemetry in plain English.

> **Important:** The custom 80.66M-parameter model primarily generates telemetry-style text; it is **not** an independently validated conversational assistant, malware detector, or APT detector. The separate English interpreter is based on Qwen3 0.6B, not on custom-trained model weights. Generated events are synthetic examples, not evidence of an attack.

## Project overview

| Category | Details |
| --- | --- |
| Dataset | DARPA Transparent Computing / CDM18 records |
| Custom architecture | Decoder-only Transformer, RoPE, GQA, RMSNorm, SwiGLU |
| Parameters | Approximately 80.66 million |
| Layers / attention | 4 layers, 8 query heads, 2 KV groups |
| Model dimension / context | 256 hidden size, 256-token context |
| Vocabulary | 151,643 tokens |
| Runtime | PyTorch for training; GGUF + Ollama for deployment |
| Observed limitations | Incomplete telemetry generations and weak natural-language question answering |

## Repository layout

- `CYBERSECURITY_SLM/src/models/` — uploaded custom Transformer model source
- `CYBERSECURITY_SLM/config/` — uploaded architecture configuration
- `CYBERSECURITY_SLM/src/training/` — instruction fine-tuning scripts from pipeline experiments
- `CYBERSECURITY_SLM/src/evaluation/` — small response comparison test
- `CYBERSECURITY_SLM/src/inference/` — illustrative two-model Ollama API workflow
- `CYBERSECURITY_SLM/data/examples/` — **small sample** Q&A dataset (not DARPA raw data)
- `CYBERSECURITY_SLM/docs/` — methods, limitations, and local setup

## Development stages

1. Prepared and tokenized cybersecurity telemetry for sequential prediction.
2. Built and trained the custom Transformer in PyTorch.
3. Exported weights and converted to GGUF with local conversion tooling.
4. Registered the custom model in Ollama as `cybersecurity-slm`.
5. Verified telemetry-style generation in terminal and Ollama desktop.
6. Ran an initial small Q&A fine-tuning experiment; outputs remained telemetry-like.
7. Tested a separate `qwen3:0.6b` interpreter in a Python Ollama workflow to obtain English explanations of generated telemetry.

## Local setup

Python 3.12 and a running Ollama service are recommended for reproducing the provided example scripts.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
ollama list
ollama run cybersecurity-slm
```

To test the *separate interpreter* after installing `qwen3:0.6b`:

```powershell
ollama pull qwen3:0.6b
python CYBERSECURITY_SLM/src/inference/english_chat_example.py
```

The training/evaluation scripts were used in an original `D:\SLM` workspace and may require restoring the root-level `model.py`, `export/config.json`, `export/tokenizer.json`, and checkpoints in that original directory. They are included for documentation and adaptation; this ZIP does not contain the original dataset or checkpoint.

## Evaluation & limitations

The custom model produces CDM18-style text, but some completions are truncated or structurally invalid. The small instruction fine-tuning pipeline completed on CPU with 22 training examples and 2 validation examples, but did **not** produce reliable English answers. The language interpreter can describe fields only when evidence is present; its output still requires validation for accuracy.

## Dataset and artifacts

The DARPA dataset, trained checkpoints, GGUF files, tokenizer, virtual environments, credentials, and model storage are intentionally excluded. Do not publish proprietary or sensitive telemetry. See `CYBERSECURITY_SLM/docs/PROJECT_NOTES.md`.

## Research direction

Improve structured CDM18 validation, compare unseen test examples, expand reviewed instruction data, evaluate hallucinations, and document reproducible conversions without conflating custom-model behavior with Qwen-generated English responses.
