# Evaluation Tools


This directory contains scripts and configuration files used to evaluate fine-tuned Whisper models on the wTIMIT dataset using standard ASR metrics. It also includes the **evaluation results from the current thesis**.

---

## Contents

- `generate_sclite_inputs.py` — Converts prediction files and references into `.ref` and `.hyp` formats for use with `sclite`.
- `config.json` — Maps experiment names to their corresponding Hugging Face model checkpoint paths or local directories.
- `WER/`, `CER/`, `MAPSSWE/` — Subdirectories containing **evaluation results from the current thesis**, broken down by metric:
  - **WER** — Word Error Rate
  - **CER** — Character Error Rate
  - **MAPSSWE** — Matched-Pair Sentence-Segment Word Error (via `sclite`), used for statistical significance testing

---

## Evaluation Workflow

1. Generate `.ref` and `.hyp` files using:

   ```bash
   python generate_sclite_inputs.py --config config.json

The script generates `.ref` and `.hyp` files for evaluating ASR models with `sclite`, using either the original Hugging Face `openai/whisper-small` model or fine-tuned checkpoints.

To evaluate your own fine-tuned models, specify their locations in a JSON configuration file passed via the `--config_file` argument. This file should map **experiment names** to the corresponding **checkpoint directories** (local paths or Hugging Face model identifiers). For example:

```json
{
  "wn_noaug": "results_eval/wn_noaug/checkpoint-1500",
  "specaugment_ld": "results_eval/specaugment_ld/checkpoint-1500",
  "aug_voiced_300_lb": "results_eval/aug_voiced_300_lb/checkpoint-1500",
  "openai_whisper_small": "openai/whisper-small"
}

