# Phone-Aware Low-Frequency Masking (PALF)

This repository contains the code and experiments supporting my MSc thesis at the University of Groningen. The project introduces **phone-aware low-frequency masking (PALF-Mask)** as a data augmentation method to improve **whispered and normal speech recognition** using fine-tuned versions of OpenAI’s Whisper model.

### 🔗 Hugging Face Model

The final model fine-tuned with the **F0-Mask** policy is available on Hugging Face:

👉 [Kokowski/whisper-small-f0-mask]([https://huggingface.co/your-model-url](https://huggingface.co/jankoko/PALF-Whisper-small))

---

## Repository Structure

```
phone-level-freq-masking/ 
├── frequency_masking/ 
│   ├── PALF-Mask.py                   # Main masking script 
│   ├── specaugment.py                 
│   └── README.md                       
├── pre_processing_helpers/ 
│   ├── check_all_phonemes.py          # Phoneme coverage checker 
│   ├── prepare_augmented_dataset.py   # Prepare masked datasets (300Hz, 1500Hz) 
│   └── README.md                      # Module documentation 
├── fine_tuning/ 
│   ├── finetune_whisper.py            # Whisper fine-tuning script 
│   └── README.md                      
├── evaluation/
│   ├── WER
│   ├── CER 
│   ├── MAPSSWE 
│   ├── evaluate_checkpoints.py        # Evaluate WER and MAPSSWE testing 
│   ├── sample_comparison.py           # Compare whisper outputs with ground truth for selected samples 
│   └── README.md                       
├── data/ 
│   ├── sample data/                   # Whisper/Normal samples
│   └── sample masked/                 # Masked files examples (300Hz, 1500Hz) 
├── README.md                          
├── LICENSE                             
├── requirements.txt
```

## What’s Included

- **PALF-Mask.py**: Implements low-frequency spectrogram masking for specific phoneme classes
- **Fine-tuning pipeline**: Full Whisper-small fine-tuning setup with support for dynamic SpecAugment
- **Evaluation**: WER, MAPSSWE, and CER scoring with reproducible output generation

---

## Status

**Documentation under construction** – All main scripts are included and functional. Detailed module-level documentation will follow shortly.
---

## License

This project is licensed under the MIT License.
                
