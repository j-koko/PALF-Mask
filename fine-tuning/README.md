# Whisper Fine-Tuning Script

This module contains the script for fine-tuning the `openai/whisper-small` model.

---

## Script: `finetune_whisper.py`

This script supports four training modes:

- `wn` — Normal + whispered training set, unaugmented
- `dup` — Duplicated training data (normal + normal) -> initally planned but not used in experiments
- `aug` — Concatenation of original and masked data (e.g., F0-masked)
- `spec` — Apply dynamic SpecAugment during training (policy configurable)

The script automatically caches preprocessed features using Hugging Face Datasets and supports fast reloading for future runs.

---

## Usage

```bash
python finetune_whisper.py \
  --train_mode aug \
  --augset plosive \
  --spec_policy LD \
  --apply_specaugment \
  --output_dir results_whisper_small
```

### Arguments

| Argument               | Type     | Required | Default                | Description                                                                 |
|------------------------|----------|----------|------------------------|-----------------------------------------------------------------------------|
| `--train_mode`         | str      | No       | `"spec"`               | Training mode: `wn`, `dup`, `aug`, or `spec`                                |
| `--augset`             | str      | No       | `"plosive"`            | Name of the augmentation set used to locate cached masked training data     |
| `--spec_policy`        | str      | No       | `"LD"`                 | SpecAugment policy: `LD`, `LB`, `SM`, `SS`, or custom                       |
| `--apply_specaugment`  | flag     | No       | `False`                | If set, applies SpecAugment dynamically during training                     |
| `--batch_size`         | int      | No       | `32`                   | Batch size used for both training and evaluation                            |
| `--output_dir`         | str      | No       | `results_whisper_small`| Directory for saving checkpoints, logs, and visualizations                  |
| `--dry_run`            | flag     | No       | `False`                | If set, performs a dry run (loads one batch, no training)                   |
| `--max_steps`          | int      | No       | `2500`                 | Maximum number of training steps                                            |

### Output

Each training run will create a dedicated subdirectory inside the specified `--output_dir`, containing:

```
<output_dir>/
└── <train_mode><augset><spec_policy>/
├── config.json # Configuration and arguments for the run
├── split_sizes.csv # Number of samples per split (original, mask, dev, test)
├── training_log.csv # Step-wise logs from the Trainer (loss, WER, etc.)
├── loss_curve.png # Plot of training loss over steps
├── dev_wer_curve.png # Plot of dev WER over steps
├── emissions.csv # (Optional) CO₂ emissions log if CodeCarbon is enabled
├── pytorch_model.bin # Final trained model weights
└── ... # Other Hugging Face model artifacts
```

> **Note:** The subdirectory is automatically named using the pattern `<train_mode>_<augset>_<spec_policy>`, e.g., `aug_plosive_ld`.

This output is fully compatible with Hugging Face’s `from_pretrained()` loading and enables reproducibility via saved configs and metrics.


### Training Modes

| Mode   | Description                                                                 |
|--------|-----------------------------------------------------------------------------|
| `wn`   | Uses the original training set (normal + whispered), no augmentation.       |
| `dup`  | Doubles the training data by concatenating it with itself (normal + normal).|
| `aug`  | Combines the original data with a preprocessed, masked version (e.g., plosive-masked). |
| `spec` | Applies dynamic SpecAugment during training using the selected policy.      |

