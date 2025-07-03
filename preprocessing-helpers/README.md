# Preprocessing and Repartitioning

This module provides a script for reorganizing and preprocessing audio data from the original NIST-style directory of wTIMIT into a partitioning compatible with E2E training. It also generates `.lab` transcription files.

---

## Script: `preprocessing.py`

This script:

- Converts the original dataset structure (`TRAIN/TEST`, `SG/US`, `normal/whisper`) into a normalized `train/dev/test` layout.
- Aligns audio files with transcriptions from a `labels.txt` file.
- Writes plain-text `.lab` files for each utterance.
- Skips problematic utterance IDs (if provided).
- Infers the `train`, `dev`, or `test` subset based on utterance IDs.
- Creates a CSV manifest (`dataset_manifest.csv`) with metadata for all successfully processed files.

---

## Expected Input

- A folder containing:
  - Audio organized in `TRAIN/` and `TEST/` directories
  - Subdirectories by mode (`normal`, `whisper`) and region (`SG`, `US`)
  - A `labels.txt` file with tab-separated transcript data, expected to include:
    - `FILE` (filename)
    - `TRANSCRIPT` (verbatim transcript)
- Optional: a text file listing problematic utterance IDs to exclude from processing

---

## Output Directory Structure

After running the script, the output directory (`--output_dir`, default: `data_split/`) will be organized as follows:

```
data_split/
├── train/
│ ├── normal/
│ │ └── US/
│ │ ├── 105/
│ │ │ ├── s105u003n.WAV
│ │ │ └── s105u003n.lab
│ │ ├── 106/
│ │ │ └── ...
│ │ └── ...
│ └── whisper/
│ └── US/
│ ├── 105/
│ │ ├── s105u004w.WAV
│ │ └── s105u004w.lab
│ ├── 106/
│ │ └── ...
│ └── ...
├── dev/
│ ├── normal/
│ └── whisper/
├── test/
│ ├── normal/
│ └── whisper/
├── dataset_manifest.csv
```
Each utterance is saved as:
- `.WAV` file (copied from original)
- `.lab` file containing the plain-text transcription

---

## Usage

```bash
python preprocessing.py \
  --data_root /path/to/nist_original \
  --labels_file labels.txt \
  --output_dir data_split \
  --problematic_ids bad_ids.txt
```

### Arguments

| Argument             | Type   | Required | Default       | Description                                                                 |
|----------------------|--------|----------|----------------|-----------------------------------------------------------------------------|
| `--data_root`        | str    | Yes      | —              | Root path to the original dataset (containing `TRAIN/`, `TEST/`, etc.).     |
| `--labels_file`      | str    | No       | `labels.txt`   | Path to the transcript list with columns `FILE` and `TRANSCRIPT`.           |
| `--output_dir`       | str    | No       | `data_split/`  | Output directory where structured audio, `.lab` files, and metadata are saved. |
| `--problematic_ids`  | str    | No       | —              | Path to a text file listing utterance IDs to skip (one per line).           |

### Notes

- This script is intended for preprocessing and repartitioning the original dataset to make it suitable for E2E training.
- Only utterances with valid transcripts (from the `labels.txt` file) are processed.
- Utterance IDs must follow the format `sXXXuYYY[n/w]`, where `XXX` is the speaker ID, `YYY` is the utterance number, and `n`/`w` indicates normal or whispered mode.
- Subset assignment (`train`, `dev`, `test`) is hardcoded based on utterance ID:
  - `utt_id` ≤ 402 → `train`
  - `403` ≤ `utt_id` ≤ `427` → `dev`
  - `428` ≤ `utt_id` ≤ `452` → `test`
- Problematic utterance IDs (e.g., corrupted or misaligned files) can be excluded by passing a plain-text list to `--problematic_ids`.
- The output directory is organized by `subset/mode/region/speaker/` and includes `.WAV` audio files, `.lab` transcripts, and a `dataset_manifest.csv`. This structure is compatible with the masking script in `low_frequency_masking/PALF-Mask.py`.


