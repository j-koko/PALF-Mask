import os
import argparse
import librosa
import numpy as np
import soundfile as sf
import textgrid
from tqdm import tqdm
from multiprocessing import Pool
import shutil

# === Defaults ===
SR = 16000
N_FFT = 512
HOP_LENGTH = 160
WIN_LENGTH = 400
DEFAULT_THRESHOLD = 300
# below constant is wTIMIT-specific (speaker dir names):
VALID_SPEAKERS = [str(s) for s in range(101, 132) if s not in (113, 114, 115)] 
# Only process train split
SPLITS_TO_PROCESS = ["train"]


    
PHONEME_GROUPS = {
    # All voiced phones in English (US) v3.0.0 MFA dictionary:
    "voiced_all": ['b', 'bʲ', 'd', 'dʲ', 'd̪', 'ɟ', 'ɟʷ', 'ɡ', 'ɡʷ', 'ɖ', 'gb', 'dʒ', 'v', 'vʲ', 'vʷ', 'ð', 'z', 'ʒ', 'm', 'mʲ', 'm̩', 'n', 'n̩', 'ŋ', 'ɲ', 'ɱ', 'j', 'w', 'ɹ', 'ʋ', 'l', 'ɫ', 'ɫ̩', 'ʎ', 'ɾ', 'ɾʲ', 'ɾ̃', 'aj', 'aw', 'i', 'iː', 'ɪ', 'ej', 'ow', 'æ', 'ə', 'ɚ', 'ɛ', 'ɝ', 'ɐ', 'ɑ', 'ɑː', 'ɒ', 'ɒː', 'ɔj', 'ʉ', 'ʉː', 'ʊ', 'a', 'aː', 'e', 'eː', 'o', 'oː', 'əw', 'ɛː', 'ɜ', 'ɜː'],
    # The voiceless phones in English (US) for completeness:
    "voiceless_all": ['p', 'pʰ', 'pʲ', 't', 'tʰ', 'tʲ', 'tʷ', 't̪', 'k', 'kʰ', 'kʷ', 'c', 'cʰ', 'cʷ', 'kp', 'ʈ', 'ʈʲ', 'ʈʷ', 'tʃ', 'f', 'fʲ', 'fʷ', 's', 'ʃ', 'ç', 'h', 'θ', 'ʔ'],
    # All US voiced phonemes used in alignement of wTIMIT in the current work (F0-Mask, F1-Mask, Hybrid):
    "voiced_wtimit": [
        'aj', 'aw', 'b', 'bʲ', 'd', 'dʒ', 'dʲ', 'd̪', 'ej', 'i', 'iː', 'j',
        'l', 'm', 'mʲ', 'm̩', 'n', 'n̩', 'ow', 'v', 'vʲ', 'w', 'z', 'æ', 'ð',
        'ŋ', 'ɐ', 'ɑ', 'ɑː', 'ɒ', 'ɒː', 'ɔj', 'ə', 'ɚ', 'ɛ', 'ɝ', 'ɟ', 'ɟʷ',
        'ɡ', 'ɡʷ', 'ɪ', 'ɫ', 'ɫ̩', 'ɲ', 'ɹ', 'ɾ', 'ɾʲ', 'ɾ̃', 'ʉ', 'ʉː', 'ʊ'],
    # All vowels used for F1-Mask:
    "vowels": [
        'aj', 'aw', 'i', 'iː', 'ɪ', 'ej', 'ow', 'æ', 'ə', 'ɚ', 'ɛ', 
        'ɝ', 'ɐ', 'ɑ', 'ɑː', 'ɒ', 'ɒː', 'ɔj', 'ʉ', 'ʉː', 'ʊ', 'a', 'aː', 'e', 
        'eː', 'o', 'oː', 'əw', 'ɛː', 'ɜ', 'ɜː'],
    # Remaining phones group by manner of articulation and voicing (for future expansion):
    "nasals": ['m', 'mʲ', 'm̩', 'n', 'n̩', 'ŋ', 'ɲ', 'ɱ'],
    "glides": ['j', 'w', 'ɹ', 'ʋ'],
    "voiced_plosives": ['b', 'bʲ', 'd', 'dʲ', 'd̪', 'ɟ', 'ɟʷ', 'ɡ', 'ɡʷ', 'ɖ', 'gb'],
    "voiceless_plosives": ['p', 'pʰ', 'pʲ', 't', 'tʰ', 'tʲ', 'tʷ', 't̪', 'k', 'kʰ', 'kʷ', 'c', 'cʰ', 'cʷ', 'kp', 'ʈ', 'ʈʲ', 'ʈʷ'],
    "voiced_fricatives": ['v', 'vʲ', 'vʷ', 'ð', 'z', 'ʒ'],
    "voiceless_fricatives": ['f', 'fʲ', 'fʷ', 's', 'ʃ', 'ç', 'h', 'θ'],
    "voiced_affricates": ['dʒ'],
    "voiceless_affricates": ['tʃ'],
    "laterals": ['l', 'ɫ', 'ɫ̩', 'ʎ'],
    "taps_flaps": ['ɾ', 'ɾʲ', 'ɾ̃']
}

def get_mask_times(textgrid_path, target_labels):
    tg = textgrid.TextGrid.fromFile(textgrid_path)
    phones_tier = tg.getFirst('phones')
    return [(intv.minTime, intv.maxTime) for intv in phones_tier if intv.mark in target_labels]

def mask_low_freq(audio, sr, mask_times, threshold, win_length, hop_length, n_fft):
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    time_bins = librosa.frames_to_time(np.arange(stft.shape[1]), sr=sr, hop_length=hop_length)
    mask_band = freqs < threshold
    mask_freqs = np.where(mask_band)[0]
    mask_frames = [i for i, t in enumerate(time_bins) if any(start <= t <= end for (start, end) in mask_times)]
    
    stft[mask_freqs[:, None], mask_frames] = 0
    return librosa.istft(stft, hop_length=hop_length, win_length=win_length)


def process_normal_file(args):
    wav_path, tg_path, lab_path, out_wav_path, out_lab_path, threshold, sr, win_length, hop_length, n_fft, masked_phones = args
    if os.path.exists(out_wav_path):
        return
    try:
        if not os.path.exists(tg_path):
            print(f"Missing TextGrid for {wav_path}")
            return

        audio, _ = librosa.load(wav_path, sr=sr)

        # Extract masking intervals
        try:
            mask_times = get_mask_times(tg_path, masked_phones)
        except Exception as e:
            print(f"Failed to parse {tg_path}: {e}")
            return


        # Check frequency mask coverage
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        time_bins = librosa.frames_to_time(np.arange(stft.shape[1]), sr=sr, hop_length=hop_length)
        mask_band = freqs < threshold
        mask_freqs = np.where(mask_band)[0]
        mask_frames = [i for i, t in enumerate(time_bins) if any(start <= t <= end for (start, end) in mask_times)]

        # Apply masking and save
        stft[mask_freqs[:, None], mask_frames] = 0
        masked_audio = librosa.istft(stft, hop_length=hop_length, win_length=win_length)
        sf.write(out_wav_path, masked_audio, sr)

        if lab_path and os.path.exists(lab_path):
            shutil.copy2(lab_path, out_lab_path)

    except Exception as e:
        print(f"Error processing {wav_path}: {e}")


def gather_tasks(input_root, output_root, threshold, sr, win_length, hop_length, n_fft, masked_phones, test_mode=False):
    """Gathering files for masking: both whisper and normal subsets of part of wTIMIT"""
    tasks = []
    count = 0
    MAX_TEST = 10

    for split in SPLITS_TO_PROCESS:
        for mode in ["normal", "whisper"]:
            for speaker in VALID_SPEAKERS:
                sp_dir = os.path.join(input_root, split, mode, "US", speaker)
                if not os.path.isdir(sp_dir):
                    continue
                for fname in os.listdir(sp_dir):
                    if not fname.lower().endswith(".wav"):
                        continue
                    base = fname[:-4]
                    wav_path = os.path.join(sp_dir, fname)
                    tg_path = os.path.join(sp_dir, base + ".TextGrid")
                    lab_path = os.path.join(sp_dir, base + ".lab")
                    rel_path = os.path.join(split, mode, "US", speaker, fname)
                    out_wav_path = os.path.join(output_root, rel_path)
                    out_lab_path = os.path.join(output_root, split, mode, "US", speaker, base + ".lab")
                    os.makedirs(os.path.dirname(out_wav_path), exist_ok=True

                    )
                    if os.path.exists(tg_path):
                        tasks.append((
                            wav_path, tg_path, lab_path,
                            out_wav_path, out_lab_path,
                            threshold, sr, win_length, hop_length, n_fft, masked_phones
                        ))
                        count += 1
                        if test_mode and count >= MAX_TEST:
                            return tasks
    return tasks


def run_all(input_root, output_root, threshold, sr, win_length, hop_length, n_fft, enabled_groups, num_workers, test_mode):
    masked_phones = set(p for group in enabled_groups for p in PHONEME_GROUPS.get(group, []))
    tasks = gather_tasks(input_root, output_root, threshold, sr, win_length, hop_length, n_fft, masked_phones, test_mode=test_mode)
    with Pool(num_workers) as pool:
        list(tqdm(pool.imap_unordered(process_normal_file, tasks), total=len(tasks)))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Apply phone-specific frequency masking to audio files based on TextGrid annotations.")

    parser.add_argument("--input_root", required=True, help="Root directory containing input audio and TextGrid files, organized by split and mode.")
    parser.add_argument("--output_root", required=True, help="Root directory where masked audio files and copied .lab files will be saved.")
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD, help=f"Frequency threshold in Hz for masking (default: {DEFAULT_THRESHOLD}). Frequencies below this will be zeroed out during masking.")
    parser.add_argument("--sr", type=int, default=SR, help=f"Sampling rate for loading audio files (default: {SR} Hz).")
    parser.add_argument("--n_fft", type=int, default=N_FFT, help=f"FFT size for STFT (default: {N_FFT}). Determines frequency resolution of the spectrogram.")
    parser.add_argument("--hop_length", type=int, default=HOP_LENGTH, help=f"Hop length (in samples) for STFT and ISTFT (default: {HOP_LENGTH}). Controls time resolution.")
    parser.add_argument("--win_length", type=int, default=WIN_LENGTH, help=f"Window length (in samples) for STFT (default: {WIN_LENGTH}).")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel worker processes for faster processing (default: 4).")
    parser.add_argument("--groups", nargs="+", default=["voiced_wtimit"], help="Phoneme groups to mask, specified as space-separated list (e.g., vowels nasals voiced_plosives). Default: voiced_wtimit.")
    parser.add_argument("--test", action="store_true", help="If set, limit the run to a small number of files for testing purposes.")

    

    args = parser.parse_args()

    run_all(
        input_root=args.input_root,
        output_root=args.output_root,
        threshold=args.threshold,
        sr=args.sr,
        win_length=args.win_length,
        hop_length=args.hop_length,
        n_fft=args.n_fft,
        enabled_groups=args.groups,
        num_workers=args.workers,
        test_mode=args.test
    )



