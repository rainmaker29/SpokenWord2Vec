'''
This script is for feature extraction from the LibriSpeech dataset using models from the s3prl library. 
The scripts supports: MFCC, HuBERT, and Wav2Vec2 models. The layer number of HuBERT base (layer 6), 
and wav2vec2 (one before the last) are hardcoded in the script. 

Requirements:
- Python libraries: os, tqdm, torch, numpy, s3prl, argparse, pickle, sklearn, librosa
- A CUDA-capable device is recommended for faster processing, although CPU is supported.
- LibriSpeech dataset: Ensure you have the LibriSpeech dataset downloaded.

Paths and Directory Structure:
- Audio files should be stored in './LibriSpeech' (Original audio files).
- Alignment files should be stored in './LibriSpeech-Alignments/LibriSpeech' (Alignment files directory).
  You can download the alignment files for LibriSpeech from https://github.com/CorentinJ/librispeech-alignments

Usage:
1. Set the feature extraction type by passing --feature_type with 'mfcc', 'wav2vec2', or 'hubert'.
3. Adjust the paths in lines 39-40 in the script if your dataset is located in a different directory.
4. Specify which dataset you want to process (e.g. train-clean, dev-clean, etc.) in line 42

Example:
python script_name.py --feature_type hubert

Features are extracted per speaker and stored in the format '[speaker_id].pkl' within the directory 'output_dir/{feature_type}/'.
'''

import os
from tqdm import tqdm
import torch
import numpy as np
import s3prl.hub as hub
import argparse
import pickle
import librosa
from sklearn.preprocessing import normalize

librispeech_root = "./LibriSpeech-Alignments/LibriSpeech"
librispeech_root_audio = "./LibriSpeech"
output_dir ="./features"
data_sets=['train-clean-100'] #''dev-clean']

# Command-line argument parsing
parser = argparse.ArgumentParser(description="Feature extraction script with multiple options.")
parser.add_argument('--feature_type', type=str, default='hubert', choices=['mfcc', 'wav2vec2', 'hubert'], 
                    help='Specify the type of features to extract (mfcc, wav2vec2, hubert).')
args = parser.parse_args()

# Set the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model selection based on command-line argument
if args.feature_type == 'mfcc':
    config_path = 's3prl/upstream/baseline/mfcc.yaml'  # Adjust the path as necessary
    model = getattr(hub, 'baseline_local', model_config=config_path).to(device)
elif args.feature_type == 'wav2vec2':
    model = getattr(hub, 'wav2vec2')().to(device)
elif args.feature_type == 'hubert':
    model = getattr(hub, 'hubert')().to(device)
else:
    raise ValueError("Unsupported feature type selected.")

model.eval()



def extract_features(audio_fpath, words, end_times):
    sample_rate = 16000
    wav, sr = librosa.load(audio_fpath, sr=sample_rate)
    wav = torch.FloatTensor(wav).to(device).unsqueeze(0)  # Reshape for batch size of 1 if needed
    words = np.array(words)
    start_times = np.array([0.0] + end_times[:-1])
    end_times = np.array(end_times)
    
    assert len(words) == len(end_times) == len(start_times)
    
    if end_times[0] < 1e-6:
        end_times[0] = 0.1
    texts = [t for t in words if t != '']
    
    with torch.no_grad():
        if args.feature_type == 'mfcc':
            features = model([wav])["hidden_states"][0]  # Adjust according to actual model output
        else:
            features = model(wav)["hidden_states"]
            layer_idx = -8 if args.feature_type == 'hubert' else -3
            features = features[layer_idx].cpu().detach().numpy()

    mfccs = []
    proportion_ratio = features.shape[1] / wav.shape[1]
    for t in range(len(start_times)):
        if words[t] != '':
            mfccs.append(features[:, int(start_times[t] * sample_rate * proportion_ratio):int(end_times[t] * sample_rate * proportion_ratio), :])

    return texts, mfccs, features

def process_speaker(speaker_id, set_name, librispeech_root, librispeech_root_audio):
    if speaker_id == '.DS_Store':
        return
    feats = []
    speaker_dir = os.path.join(librispeech_root, set_name, speaker_id)
    speaker_dir_audio = os.path.join(librispeech_root_audio, set_name, speaker_id)
    for book_id in os.listdir(speaker_dir):
        if book_id == '.DS_Store':
            continue
        book_dir = os.path.join(speaker_dir, book_id)
        book_dir_audio = os.path.join(speaker_dir_audio, book_id)
        alignment_fpath = os.path.join(book_dir, f"{speaker_id}-{book_id}.alignment.txt")
        if not os.path.exists(alignment_fpath):
            raise FileNotFoundError(f"Alignment file not found: {alignment_fpath}")
        with open(alignment_fpath, "r") as alignment_file:
            for line in alignment_file:
                try:
                    utterance_id, words, end_times = line.strip().split(' ')
                    words = words.replace('\"', '').split(',')
                    end_times = [float(e) for e in end_times.replace('\"', '').split(',')]
                    audio_fpath = os.path.join(book_dir_audio, f"{utterance_id}.flac")
                    texts, mfccs, features = extract_features(audio_fpath, words, end_times)
                    for t, mfcc_entry in zip(texts, mfccs):
                        feats.append([mfcc_entry, t])
                    feats.append(["\n", "\n"])
                except Exception as e:
                    print(f"Error processing {line}: {e}")
        os.makedirs(os.path.join(output_dir, f"{args.feature_type}/speakers"), exist_ok=True)
        with open(os.path.join(output_dir, f"{args.feature_type}/speakers/{speaker_id}.pkl"), "wb") as f:
            pickle.dump(feats, f)


if __name__ == "__main__":
    for fidx, set_name in enumerate(data_sets):
        print("Processing folder:", set_name)
        if not os.path.isdir(os.path.join(librispeech_root, set_name)):
            print(f"Directory not found: {set_name}")
            continue
        speaker_ids = os.listdir(os.path.join(librispeech_root, set_name))
        
        for speaker_id in tqdm(speaker_ids):
                process_speaker(speaker_id, set_name, librispeech_root, librispeech_root_audio)
