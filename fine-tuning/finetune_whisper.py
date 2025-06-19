#!/usr/bin/env python

import argparse, os, json, shutil, pathlib, csv
import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
from datetime import datetime
from datasets import load_from_disk, concatenate_datasets, Audio
from torch.utils.data import IterableDataset
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    Seq2SeqTrainer, Seq2SeqTrainingArguments
)
import evaluate
from whisper_normalizer.english import EnglishTextNormalizer

# CO2_Tracking disabled by default
try:
    from codecarbon import EmissionsTracker
    CO2_TRACKING = False # change to True to enable CO2 emission tracking
except ImportError:
    CO2_TRACKING = False
  
# Random seed across fine-tuning runs to enable replicability
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

class ShuffleAllDataset(torch.utils.data.IterableDataset):
    def __init__(self, hf_dataset, seed=0):
        self.ds = hf_dataset
        self.base_seed = seed
    def __iter__(self):
        rng = np.random.default_rng(self.base_seed + int(torch.randint(0, 1 << 30, (1,))))
        idx = np.arange(len(self.ds))
        rng.shuffle(idx)
        for i in idx:
            yield self.ds[int(i)]
    def __len__(self):
        return len(self.ds)

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune Whisper-small with SpecAugment and CO2 logging.")
    p.add_argument("--train_mode", choices=["wn", "dup", "aug", "spec"], default="spec")
    p.add_argument("--augset", type=str, default="plosive")
    p.add_argument("--spec_policy", type=str, default="LD", help="SpecAugment policy: LD, LB, SM, SS, or custom")
    p.add_argument("--apply_specaugment", action="store_true", help="Apply SpecAugment dynamically during training")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--output_dir", type=str, default="results_whisper_small")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--max_steps", type=int, default=2500)
    return p.parse_args()

def save_run_config(run_dir, args_dict):
    config_path = os.path.join(run_dir, "config.json")
    config = {
        "run_timestamp": datetime.now().isoformat(),
        "config": vars(args_dict)
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

def main():
    args = parse_args()
    torch.backends.cudnn.benchmark = True
    ROOT_ORIG = "us_normal_and_whisper"
    CACHE_DIR = pathlib.Path("cache_prepared"); CACHE_DIR.mkdir(exist_ok=True)

    ds1 = load_from_disk(ROOT_ORIG)
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")

    def load_or_prepare(name: str, hf_ds):
        cache_path = CACHE_DIR / name
        if cache_path.is_dir():
            return load_from_disk(str(cache_path))
        mapped = hf_ds.map(
            lambda b: {
                "input_features": processor(b["path"]["array"], sampling_rate=b["path"]["sampling_rate"], return_tensors="pt").input_features[0],
                "labels": processor.tokenizer(b["text"]).input_ids
            },
            remove_columns=hf_ds.column_names,
            num_proc=4,
            desc=f"Mapping {name}",
        )
        mapped.save_to_disk(str(cache_path))
        return mapped
        
    def load_or_prepare_mask(name: str, raw_ds):
        """
        Create (or re-use) a cache that already contains
        input_features  +  labels  for the *masked* dataset.
        `name` will be something like  train_mask_300  or  train_mask_plosive.
        """
        cache_path = CACHE_DIR / name
        if cache_path.is_dir():
            return load_from_disk(str(cache_path))

        print(f"… Creating masked cache {cache_path}")
        mapped = raw_ds.map(
            lambda b: {
                "input_features": processor(
                    b["path"]["array"],
                    sampling_rate=b["path"]["sampling_rate"],
                    return_tensors="pt"
                ).input_features[0],
                "labels": processor.tokenizer(b["text"]).input_ids
            },
            remove_columns=raw_ds.column_names,
            num_proc=4,
            desc=f"Mapping {name}",
        )
        mapped.save_to_disk(str(cache_path))
        return mapped

    train_orig = load_or_prepare("train_orig", ds1["train"])
        
    if args.train_mode == "aug":
        prepared_mask_path = CACHE_DIR / f"train_mask_{args.augset}"
        if not prepared_mask_path.exists():
            raise FileNotFoundError(f"Expected cached masked dataset at {prepared_mask_path}")
        train_mask = load_from_disk(str(prepared_mask_path))
    else:
        train_mask = None

    if args.train_mode == "wn":
        pool = train_orig
    elif args.train_mode == "dup":
        pool = concatenate_datasets([train_orig, train_orig])
    elif args.train_mode == "aug":
        pool = concatenate_datasets([train_orig, train_mask])

    else:
        pool = train_orig

    train_ds = ShuffleAllDataset(pool, seed=42)
    dev_ds = load_or_prepare("dev_prepared", ds1["dev"].cast_column("path", Audio(decode=True)))
    test_ds = load_or_prepare("test_prepared", ds1["test"].cast_column("path", Audio(decode=True)))
    
    
    def save_split_sizes(run_dir, train_orig, dev_ds, test_ds, train_mask=None):
        split_info = {
            "train_orig": len(train_orig),
            "train_mask": len(train_mask) if train_mask is not None else 0,
            "dev": len(dev_ds),
            "test": len(test_ds),
        }

        print(f"► Train  orig : {split_info['train_orig']}")
        if train_mask is not None:
            print(f"► Train  mask : {split_info['train_mask']}")
        print(f"► Dev         : {split_info['dev']}")
        print(f"► Test        : {split_info['test']}\n")
        # --- Save as CSV ---
        csv_path = pathlib.Path(run_dir) / "split_sizes.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["split", "count"])
            for key, val in split_info.items():
                writer.writerow([key, val])
        print(f"Split sizes saved to:{csv_path}")
        

    

    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")
    model.config.dropout = 0.1
    model.config.attention_dropout = 0.1
    model.config.activation_dropout = 0.1
    model.config.encoder_layerdrop = 0.0
    model.config.decoder_layerdrop = 0.0

    pad_id = processor.tokenizer.pad_token_id

    specaugment_transform = None
    if args.apply_specaugment:
        import torchaudio.transforms as T
        from specaugment import specaugment as custom_specaugment
        if args.spec_policy.lower() in ["lb", "ld", "sm", "ss"]:
            def build_specaugment(policy):
                if policy in ["ld", "lb", "ss", "sm"]:
                    time_param = 100 if policy in ["ld", "lb"] else 70
                    freq_param = 27 if policy in ["ld", "lb", "ss"] else 15
                    time_masks = 1 if policy == "lb" else 2
                    freq_masks = 1 if policy == "lb" else 2

                    def transform(spec):
                        for _ in range(time_masks):
                            spec = T.TimeMasking(time_mask_param=time_param)(spec)
                        for _ in range(freq_masks):
                            spec = T.FrequencyMasking(freq_mask_param=freq_param)(spec)
                        return spec
                    return transform
                else:
                    from specaugment import specaugment as custom_specaugment
                    return lambda x: custom_specaugment(x, policy=policy)
            
        else:
            specaugment_transform = build_specaugment(args.spec_policy.lower())

    def collate(batch):
            feats = []
            for b in batch:
                x = torch.tensor(b["input_features"])
                if specaugment_transform and trainer.model.training:
                    x = x.T
                    x = specaugment_transform(x)
                    x = x.T
                feats.append(x)
            X = torch.stack(feats)
            y = [torch.tensor(b["labels"]) for b in batch]
            y = torch.nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=pad_id)
            y[y == pad_id] = -100
            return {"input_features": X, "labels": y}


    normalizer = EnglishTextNormalizer()
    wer_metric = evaluate.load("wer")
    def compute_metrics(out):
        """Validation WER computation with evaluate"""
        preds = processor.batch_decode(out.predictions, skip_special_tokens=True)
        labels = np.where(out.label_ids != -100, out.label_ids, pad_id)
        refs = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
        preds = [normalizer(p) for p in preds]
        refs = [normalizer(r) for r in refs]
        return {"dev_wer": wer_metric.compute(predictions=preds, references=refs)}

    run_dir = os.path.join(args.output_dir, f"{args.train_mode}_{args.augset}_{args.spec_policy.lower()}")
    os.makedirs(run_dir, exist_ok=True)
    save_run_config(run_dir, args)
    
    save_split_sizes(run_dir, train_orig, dev_ds, test_ds, train_mask)

    emissions_tracker = None
    if CO2_TRACKING:
        # CO2 emissions will be logged to <run_dir>/emissions.csv
        emissions_tracker = EmissionsTracker(output_dir=run_dir, output_file="emissions.csv")
        emissions_tracker.start()

    training_args = Seq2SeqTrainingArguments(
        output_dir=run_dir,
        seed=SEED,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=1e-6,
        weight_decay=0.0,
        max_steps=args.max_steps,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=1,
        warmup_steps=10,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        predict_with_generate=True,
        report_to="none",
        metric_for_best_model="dev_wer",
        greater_is_better=False,
        load_best_model_at_end=True,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=processor.feature_extractor,
        data_collator=collate,
        compute_metrics=compute_metrics,
    )

    if args.dry_run:
        batch = next(iter(trainer.get_train_dataloader()))
        print("✓ dry-run batch shapes:", batch["input_features"].shape, batch["labels"].shape)
        return
        
    trainer.model.config.save_safetensors = False

    trainer.train()
    trainer.save_model(run_dir)

    if emissions_tracker:
        emissions_tracker.stop()

    log_df = pd.DataFrame(trainer.state.log_history)
    log_df.to_csv(os.path.join(run_dir, "training_log.csv"), index=False)

    if "loss" in log_df.columns:
        plt.figure(); plt.plot(log_df["step"], log_df["loss"], marker="o")
        plt.xlabel("Training step"); plt.ylabel("Loss")
        plt.title("Training loss vs. step"); plt.grid(True, ls="--", lw=0.5)
        plt.tight_layout(); plt.savefig(os.path.join(run_dir, "loss_curve.png")); plt.close()

    if "dev_wer" in log_df.columns:
        wer_df = log_df[log_df["dev_wer"].notnull()]
        if not wer_df.empty:
            plt.figure(); plt.plot(wer_df["step"], wer_df["dev_wer"], marker="o")
            plt.xlabel("Training step"); plt.ylabel("WER")
            plt.title("Validation (Dev) WER vs. step"); plt.grid(True, ls="--", lw=0.5)
            plt.tight_layout(); plt.savefig(os.path.join(run_dir, "dev_wer_curve.png")); plt.close()

if __name__ == "__main__":
    main()