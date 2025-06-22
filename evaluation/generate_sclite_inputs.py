import os
import json
import torch
from datasets import load_from_disk, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
from whisper_normalizer.english import EnglishTextNormalizer
from tqdm import tqdm
import argparse

def prepare_example(example, processor):
    audio = example["path"]
    example["input_features"] = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features[0]
    example["labels"] = processor.tokenizer(example["text"]).input_ids
    example["utt_id"] = os.path.splitext(os.path.basename(audio["path"]))[0]
    return example

def custom_collate_fn(features, pad_id):
    input_features = torch.stack([torch.tensor(f["input_features"]) for f in features])
    label_list = [torch.tensor(f["labels"]) for f in features]
    labels_padded = torch.nn.utils.rnn.pad_sequence(label_list, batch_first=True, padding_value=pad_id)
    labels_padded[labels_padded == pad_id] = -100
    return {"input_features": input_features, "labels": labels_padded}

def generate_predictions(checkpoint_path, experiment_id, input_root, subset, output_root, processor, normalizer, pad_id):
    print(f"\nLoading {subset} test set for experiment: {experiment_id}...")
    raw = load_from_disk(input_root)
    raw = raw.cast_column("path", Audio(decode=True))
    test_set = raw.filter(lambda x: "w" in x["filename"]) if subset == "whisper" else raw.filter(lambda x: "w" not in x["filename"])
    test_set = test_set.map(lambda ex: prepare_example(ex, processor), remove_columns=test_set.column_names)

    model = WhisperForConditionalGeneration.from_pretrained(checkpoint_path)
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None

    trainer = Seq2SeqTrainer(
        model=model,
        data_collator=lambda f: custom_collate_fn(f, pad_id),
        args=Seq2SeqTrainingArguments(
            output_dir="temp_out",
            per_device_eval_batch_size=8,
            remove_unused_columns=False,
            predict_with_generate=True,
            report_to="none",
        )
    )

    use_labels = not checkpoint_path.startswith("openai/")
    preds = trainer.predict(test_set)

    pred_strs = processor.batch_decode(preds.predictions, skip_special_tokens=True)
    pred_strs = [normalizer(p) for p in pred_strs]
    utt_ids = [ex["utt_id"] for ex in test_set]

    if use_labels:
        label_strs = processor.tokenizer.batch_decode(preds.label_ids, skip_special_tokens=True)
        label_strs = [normalizer(r) for r in label_strs]
    else:
        label_strs = [ex["text"] for ex in test_set]

    os.makedirs(output_root, exist_ok=True)
    ref_file = os.path.join(output_root, f"{experiment_id}_{subset}.ref")
    hyp_file = os.path.join(output_root, f"{experiment_id}_{subset}.hyp")

    with open(ref_file, "w") as ref_f, open(hyp_file, "w") as hyp_f:
        for uid, ref, hyp in zip(utt_ids, label_strs, pred_strs):
            ref_f.write(f"{ref} ({uid})\n")
            hyp_f.write(f"{hyp} ({uid})\n")

    print(f"Saved: {ref_file}, {hyp_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True, help="Path to JSON config mapping experiment names to checkpoint paths")
    parser.add_argument("--input_root", required=True, help="Path to test set directory")
    parser.add_argument("--output_dir", required=True, help="Directory to save .ref and .hyp files")
    parser.add_argument("--subsets", nargs="+", choices=["whisper", "normal"], default=["whisper"], help="Which subsets to process")
    parser.add_argument("--experiments", nargs="+", help="Optional list of experiment names to run (default: all from config)")
    args = parser.parse_args()

    with open(args.config_file) as f:
        config = json.load(f)

    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    normalizer = EnglishTextNormalizer()
    pad_id = processor.tokenizer.pad_token_id

    exp_names = args.experiments if args.experiments else list(config.keys())
    for exp_name in exp_names:
        if exp_name not in config:
            print(f"Skipping unknown experiment: {exp_name}")
            continue
        for subset in args.subsets:
            generate_predictions(
                checkpoint_path=config[exp_name],
                experiment_id=exp_name,
                input_root=args.input_root,
                subset=subset,
                output_root=args.output_dir,
                processor=processor,
                normalizer=normalizer,
                pad_id=pad_id
            )