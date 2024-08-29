import os
import pickle
from tqdm import tqdm
import torch
import argparse

# Command-line argument parsing
parser = argparse.ArgumentParser(description="Feature processing script.")
parser.add_argument('--feature_type', type=str, default='hubert', choices=['mfcc', 'wav2vec2', 'hubert'], 
        help='Specify the type of features to process (mfcc, wav2vec2, hubert).')
args = parser.parse_args()



output_dir =os.path.join("./features/", f"{args.feature_type}")
# Directory where the feature files are stored
feats_dir = os.path.join(output_dir, "speakers")

# Lists to hold the words and their corresponding feature vectors
wv = []
words = []
all_words = []


for file_name in tqdm(os.listdir(feats_dir)):
      file_path = os.path.join(feats_dir, file_name)
      if os.path.isfile(file_path):
          with open(file_path, 'rb') as fin2:
              data = pickle.load(fin2)
              for feat in data:
                  if feat[1] != '\n':
                      wv.append(torch.tensor(feat[0]))
                      words.append(feat[1])
                  all_words.append(feat[1])


# Serialize the collected data into pickle files
with open(os.path.join(output_dir, "all_words.pkl"), "wb") as f:
    pickle.dump(all_words, f)

with open(os.path.join(output_dir,"words.pkl"), "wb") as f:
    pickle.dump(words, f)

with open(os.path.join(output_dir,"feats.pkl"), "wb") as f:
    pickle.dump(wv, f)

print("Data has been processed and serialized into pickle files.")


