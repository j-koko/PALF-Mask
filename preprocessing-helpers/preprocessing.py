import os
import pandas as pd
from pathlib import Path
import shutil
import argparse

def load_problematic_ids(file_path):
    if not file_path:
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def determine_subset(utt_id):
    if utt_id <= 402:
        return 'train'
    elif 403 <= utt_id <= 427:
        return 'dev'
    elif 428 <= utt_id <= 452:
        return 'test'
    return None

def main(args):
    problematic_ids = load_problematic_ids(args.problematic_ids)
    labels_df = pd.read_csv(args.labels_file, sep='\t')
    labels_df['id'] = labels_df['FILE'].apply(lambda x: Path(x).stem)
    labels_df = labels_df.rename(columns={'TRANSCRIPT': 'text'})
    labels_df['utt_id'] = labels_df['id'].apply(lambda x: int(x.split('u')[1][:-1]))

    rows = []
    for split in ['TRAIN', 'TEST']:
        for mode in ['normal', 'whisper']:
            for region in ['SG', 'US']:
                audio_dir = Path(args.data_root) / split / mode / region
                if not audio_dir.exists():
                    continue
                for speaker_folder in audio_dir.glob('*'):
                    for wav_file in speaker_folder.glob('*.WAV'):
                        file_id = wav_file.stem
                        if file_id in problematic_ids:
                            continue
                        match = labels_df[labels_df['id'] == file_id]
                        if match.empty:
                            print(f"No transcript for {file_id}, skipping.")
                            continue
                        text = match.iloc[0]['text']
                        utt_id = int(file_id.split('u')[1][:-1])
                        subset = determine_subset(utt_id)
                        if subset is None:
                            print(f"Unknown subset for {file_id}, skipping.")
                            continue
                        # Target structure
                        target_dir = Path(args.output_dir) / subset / mode / region / speaker_folder.name
                        target_dir.mkdir(parents=True, exist_ok=True)
                        target_wav = target_dir / wav_file.name
                        target_lab = target_wav.with_suffix('.lab')
                        shutil.copy(wav_file, target_wav)
                        with open(target_lab, 'w', encoding='utf-8') as f:
                            f.write(text.strip())
                        # Append metadata row
                        rows.append({
                            "path": str(target_wav.resolve()),
                            "filename": file_id,
                            "speaker": speaker_folder.name,
                            "mode": mode,
                            "region": region,
                            "subset": subset,
                            "text": text.strip()
                        })

    # Save metadata as a manifest CSV
    manifest_df = pd.DataFrame(rows)
    manifest_df.to_csv(Path(args.output_dir) / "dataset_manifest.csv", index=False)
    print(f"Processing complete! Metadata saved to {Path(args.output_dir) / 'dataset_manifest.csv'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess dataset with full metadata and .lab files.")
    parser.add_argument("--data_root", type=str, required=True, help="Root path to the nist dataset.")
    parser.add_argument("--labels_file", type=str, default="labels.txt", help="Path to the labels file.")
    parser.add_argument("--output_dir", type=str, default="data_split", help="Destination directory for output data.")
    parser.add_argument("--problematic_ids", type=str, help="Path to file listing problematic utterance IDs.")
    args = parser.parse_args()
    main(args)
