# phone-level-frequency-masking
A repository for experiments and code used in my MSc thesis at University of Groningen, focused on phoneme-level frequency masking for whispered and normal speech recognition.

# file structure
phoneme-frequency-masking/
├── phone_level_freq_masking/
│   ├── low_frequency_masking.py       # Main masking script
│   ├── specaugment.py                 # Manual SpecAugment implementation
│   └── README.md                      # Module documentation
├── pre_processing_helpers/
│   ├── check_all_phonemes.py          # Phoneme coverage checker
│   ├── prepare_augmented_dataset.py   # Prepare masked datasets (300Hz, 1500Hz)
│   └── README.md                      # Module documentation
├── fine_tuning/
│   ├── finetune_whisper.py            # Whisper fine-tuning script
│   └── README.md                      # Module documentation
├── evaluation/
│   ├── evaluate_checkpoints.py        # Evaluate WER and MAPSSWE testing
│   ├── sample_comparison.py           # Compare ASR outputs for selected samples
│   └── README.md                      # Module documentation
├── data/
│   ├── (Optional) sample data/        # Example datasets for quick testing
│   └── (Optional) masked/             # Masked datasets (300Hz, 1500Hz)
├── README.md                          # Main documentation
├── LICENSE                            # License information
├── requirements.txt                   # Python dependencies
