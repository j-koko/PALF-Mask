# Generate Sclite Inputs

This script generates `.ref` and `.hyp` files for ASR model evaluation with `sclite`, using Hugging Face Whisper model or fine-tuned checkpoints.

## Configuration File

The script expects a JSON config file via the `--config_file` argument.

This file should map **experiment names** to the **path to the model checkpoint directory**, for example:

```json
{
  "wn_noaug": "results_eval/wn_noaug/checkpoint-1500",
  "specaugment_ld": "results_eval/specaugment_ld/checkpoint-1500",
  "aug_voiced_300_lb": "results_eval/aug_voiced_300_lb/checkpoint-1500",
  "openai_whisper_small": "openai/whisper-small"
}
