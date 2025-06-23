# Phone-Aware Low-Frequency Masking (PALF)

This repository contains the code and experiments supporting my MSc thesis at the University of Groningen. The project introduces **phone-aware low-frequency masking (PALF-Mask)** as a data augmentation method to improve **whispered and normal speech recognition** using fine-tuned versions of OpenAI’s Whisper model.

### 🔗 Hugging Face Models

[F0-Mask](https://huggingface.co/jankoko/PALF-Whisper-small) • 
[F1-Mask](https://huggingface.co/jankoko/PALF-F1-Whisper-small) • 
[LF-Mask](https://huggingface.co/jankoko/PALF-LF-Whisper-small) • 
[SpecAugment (baseline)](https://huggingface.co/jankoko/SpecAugment-Whisper-small)

---

## Repository Structure

```
PALF-Mask/
├── low_frequency_masking/               
│   └── PALF-Mask.py                     # Main masking script implementing F0/F1/LF masking
├── pre_processing_helpers/             # Data preprocessing utilities
│   └── preprocessing.py                # Loads wTIMIT audio and transcriptions, cleans, splits into train/dev/test
├── fine_tuning/
│   └── finetune_whisper.py             # Whisper fine-tuning pipeline with Hugging Face Trainer
├── evaluation/
│   ├── WER/, CER/, MAPSSWE/            # Evaluation outputs per metric
│   ├── config.json                     # Mapping of experiment names to model checkpoints
│   ├── generate_sclite_inputs.py       # Generate `.ref` and `.hyp` files for sclite evaluation
│   └── README.md
├── CITATION.cff # Machine-readable citation metadata (for GitHub)
├── requirements.txt
├── LICENSE
└── README.md
```

## What’s Included

- **PALF-Mask Implementation**: Low-frequency spectrogram masking applied to specific phonemes below defined frequency cutoffs (F0-Mask, F1-Mask, LF-Mask).
- **Whisper Fine-tuning Pipeline**: Custom training script using the Hugging Face Trainer.
- **Evaluation Results**: WER, CER, and MAPSSWE scores generated using the `sclite` toolkit.
- **Preprocessing Utilities**: Tools for wTIMIT-specific audio and transcript loading, cleaning, and train/dev/test split creation.
- **Prediction and Evaluation Tools**: Utilities for generating predictions and references across multiple models using JSON configuration, with outputs prepared for `sclite` evaluation.

## Status

Documentation under construction - Main scripts are included and functional. Detailed module-level documentation will follow shortly.

## License

This project is licensed under the MIT License.

## Cite this work

If you use this code or model, please cite the accompanying thesis:

> Kokowski, J. (2025). *F0-Based Masking Policies for Self-Supervised Whispered Speech Recognition*. Master’s Thesis, University of Groningen, Campus Fryslân.  
> Available at: [https://campus-fryslan.studenttheses.ub.rug.nl/id/eprint/674](https://campus-fryslan.studenttheses.ub.rug.nl/id/eprint/674)
              
