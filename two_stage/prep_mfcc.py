import tensorflow as tf
import os
from tqdm import tqdm
import torchaudio
import sounddevice as sd
import numpy as np
import librosa
import os
import warnings
import torch
import speechpy
from tqdm import tqdm
import soundfile as sf
import pickle
import torch
import s3prl.hub as hub

device = 'cuda' # or cpu
config_path = 's3prl/s3prl/upstream/baseline/mfcc.yaml'
extracter = getattr(hub, 'baseline_local')(model_config=config_path).to(device)

wavs = [torch.zeros(160000, dtype=torch.float).to(device) for _ in range(16)]
with torch.no_grad():
    mfcc = extracter(wavs)["hidden_states"]
warnings.filterwarnings("ignore")
librispeech_root = "../LibriSpeech_AG"    # Replace with yours
librispeech_root_audio = "../LibriSpeech_OG"    # Replace with yours

import torch
import s3prl.hub as hub

device = 'cuda' # or cpu
config_path = 's3prl/s3prl/upstream/baseline/mfcc.yaml'
extracter = getattr(hub, 'baseline_local')(model_config=config_path).to(device)


def save_just_word(audio_fpath, words, end_times):
    words = np.array(words)
    texts = [t for t in words if t!='']
    return texts
def split_on_silences(audio_fpath, words, end_times):
    # Load the audio waveform
    sample_rate = 16000     # Sampling rate of LibriSpeech 
    win_lenth = 0.032  #windows length for mfccs, 25ms of signal (400 ffts)
    win_shift = 0.010 #10ms windows shift
    wav, sr = librosa.load(audio_fpath, sr=sample_rate)
    words = np.array(words)
    start_times = np.array([0.0] + end_times[:-1])
    end_times = np.array(end_times)
    
    assert len(words) == len(end_times) == len(start_times) 
    if end_times[0] < 1e-6:
        end_times[0] = 0.1
    texts = [t for t in words if t!='']
    
    mfccs=[]
    
    with torch.no_grad():
        mfcc_concat = extracter([torch.tensor(wav)])["hidden_states"][0]
        for t in range(len(start_times)):
         if words[t]!='':
            proportion_ratio = mfcc_concat.shape[1]/len(wav)
            mfccs.append(mfcc_concat[:,(start_times[t]*sample_rate*proportion_ratio).astype(np.int):(end_times[t]*sample_rate*proportion_ratio).astype(np.int),:])
    return  texts, mfccs,mfcc_concat#,wavs

tot_mfcc = []
unique_word_id = 0
for fidx,set_name in enumerate(['train-clean-100']):#,'dev-clean-100',
    print("folder :",set_name)
    set_dir = os.path.join(librispeech_root, set_name)
    set_dir_audio = os.path.join(librispeech_root_audio, set_name)
    if not os.path.isdir(set_dir):
        continue
    
    for speaker_id in tqdm(os.listdir(set_dir)):
        feats = list()
        if speaker_id == '.DS_Store':
            continue
        speaker_dir = os.path.join(set_dir, speaker_id)
        speaker_dir_audio = os.path.join(set_dir_audio, speaker_id)
        # Select books
        for book_id in (os.listdir(speaker_dir)):
            idx=0
            if book_id == '.DS_Store':
                continue
            book_dir = os.path.join(speaker_dir, book_id)
            book_dir_audio = os.path.join(speaker_dir_audio, book_id)            
            # Get the alignment file
            alignment_fpath = os.path.join(book_dir, "%s-%s.alignment.txt" % 
                                            (speaker_id, book_id))
            if not os.path.exists(alignment_fpath):
                raise Exception("Alignment file not found. Did you download and merge the txt "
                                "alignments with your LibriSpeech dataset?")
            # Parse each utterance present in the file
            alignment_file = open(alignment_fpath, "r")
            for line in (alignment_file):
                utterance_id, words, end_times = line.strip().split(' ')
#                 print(words)
#                 break
                words = words.replace('\"', '').split(',')
                end_times = [float(e) for e in end_times.replace('\"', '').split(',')]
                audio_fpath = os.path.join(book_dir_audio, utterance_id + '.flac')
                texts, mfccs,mfcc_concat = split_on_silences(audio_fpath, words, end_times)
                for t,mfcc_entry in zip(texts,mfccs):
                    feats.append([mfcc_entry.cpu().detach().numpy(),t])
                    unique_word_id+=1
                feats.append(["\n","\n"])
                    
                idx+=1
            alignment_file.close()
        with open("/l/users/mohammad.sayeed/scp_files_train_clean_s3prl/"+str(speaker_id)+".scp","wb") as f:
            pickle.dump(feats,f)
            tot_mfcc.append(feats)

from sklearn.preprocessing import normalize
def generate_training_data(tokens, window_size,mfcc):
    N = len(tokens)
    X, Y = [], []
    for i in range(N):
        nbr_inds = list(range(max(0, i - window_size), i)) + \
                   list(range(i + 1, min(N, i + window_size + 1)))
        for j in nbr_inds:
            X.append(normalize(mfcc[i],axis=1))
            Y.append(normalize(mfcc[j],axis=1))     
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

from pathlib import Path
# test_other_scp = os.listdir("./scp_files_test_other")
test_clean_scp = os.listdir("/l/users/mohammad.sayeed/scp_files_train_clean_s3prl")
# create a Path object with the path to the file
with open("/l/users/mohammad.sayeed/scp_files_train_clean_s3prl/train_clean_100_s3prl.scp","w") as f:
    for t in test_clean_scp:
        if Path('/l/users/mohammad.sayeed/scp_files_train_clean_s3prl/'+t).is_file():
            if "t" in t:
                continue
            f.write(t)
            f.write("\n")