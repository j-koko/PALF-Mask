# Phone-Aware-Low-Frequency-Masking
A repository for experiments and code used in my MSc thesis at University of Groningen, focused on phone-aware low-frequency masking as an data augmentation method for whispered and normal speech recognition.

# file structure
```
phone-level-freq-masking/ 
├── frequency_masking/ 
│   ├── low_frequency_masking.py       # Main masking script 
│   ├── specaugment.py                 # Manual SpecAugment implementation 
│   └── README.md                       
├── pre_processing_helpers/ 
│   ├── check_all_phonemes.py          # Phoneme coverage checker 
│   ├── prepare_augmented_dataset.py   # Prepare masked datasets (300Hz, 1500Hz) 
│   └── README.md                      # Module documentation 
├── fine_tuning/ 
│   ├── finetune_whisper.py            # Whisper fine-tuning script 
│   └── README.md                      
├── evaluation/ 
│   ├── evaluate_checkpoints.py        # Evaluate WER and MAPSSWE testing 
│   ├── sample_comparison.py           # Compare whisper outputs with ground truth for selected samples 
│   └── README.md                       
├── data/ 
│   ├── sample data/                   # Whisper/Normal samples
│   └── sample masked/                 # Masked files examples (300Hz, 1500Hz) 
├── README.md                          
├── LICENSE                             
├── requirements.txt                   
