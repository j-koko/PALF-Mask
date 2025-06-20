# Phone-Aware Low-Frequency Masking (PALF)

This repository contains the code and experiments supporting my MSc thesis at the University of Groningen. The project introduces **phone-aware low-frequency masking (PALF-Mask)** as a data augmentation method to improve **whispered and normal speech recognition** using fine-tuned versions of OpenAI’s Whisper model.

### 🔗 Hugging Face Model

The final model fine-tuned with the **F0-Mask** policy is available on Hugging Face:

👉 [Kokowski/whisper-small-f0-mask](https://huggingface.co/jankoko/PALF-Whisper-small)

---

## Repository Structure

```
PALF-Mask/
├── low_frequency_masking/ # Core augmentation methods
│ ├── PALF-Mask.py # Main masking script
│ └── 
├── pre_processing_helpers/ # Data preprocessing utilities
│ ├── 
│ └── 
├── fine_tuning/
│ └── finetune_whisper.py # Whisper fine-tuning pipeline
├── evaluation/
│ ├── 
│ ├── WER/, CER/, MAPSSWE/ # Evaluation outputs
├── requirements.txt
├── LICENSE
└── README.md
```

## What’s Included

- **PALF-Mask.py**: Implements low-frequency spectrogram masking for specific phoneme classes
- **Fine-tuning pipeline**: Full Whisper-small fine-tuning setup with support for dynamic SpecAugment
- **Evaluation**: WER, MAPSSWE, and CER scoring with reproducible output generation

---

## Status

Documentation under construction - Main scripts are included and functional. Detailed module-level documentation and preprocessing tools will follow shortly.
---

## License

This project is licensed under the MIT License.

## Cite this work

If you use this code or model, please cite the accompanying thesis:

> Kokowski, J. (2025). *F0-Based Masking Policies for Self-Supervised Whispered Speech Recognition*. Master’s Thesis, University of Groningen, Campus Fryslân. **(To appear)**

The thesis will be publicly available soon at:  
[https://campus-fryslan.studenttheses.ub.rug.nl/view/degree_programme/voice=5Ftechnology.html](https://campus-fryslan.studenttheses.ub.rug.nl/view/degree_programme/voice=5Ftechnology.html)  
Please cite once it is available.
              
