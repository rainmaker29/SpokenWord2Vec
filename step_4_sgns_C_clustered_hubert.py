'''
dependencies: nltk, pandas, Levenshtein, gensim
'''

import io
import logging
import math
import os
import pickle
import random
import string
import sys

import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from Levenshtein import distance as edit_distance

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import FloatTensor as FT
from torch.utils.data import Dataset, DataLoader
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from gensim.models import KeyedVectors, Word2Vec
from gensim.test.utils import get_tmpfile

from tqdm import tqdm

from collections import defaultdict

import utils
import datasets
import models
from datasets import AudioDataset

from numpy.linalg import norm
from pandas.core.common import flatten
from gensim import models
import pandas as pd
from utils import add_suffix_to_lists_of_words
from utils import p_editDist, p_cosine, cosine, normalize, calc_test_sim, calc_test_dist, cal_test_edit, create_test_set


feats_dir = "./features/hubert"
output_dir = "./output/hubert_s"

text_filename ='librispeech_100.txt' #TODO

#TODO model_name
s_scale = 1
model_name = 'sgns_C_HuBERT_k100_normalized_s'
input_dim = 102
all_words_file = os.path.join(feats_dir,"all_words.pkl")
words_file = os.path.join(feats_dir,"words.pkl")
wv_file = os.path.join(feats_dir,"kmeans_100_cids.pkl")
resume_training = True

if len(sys.argv) < 2:
    print("Usage: python script.py <s_scale>")
    sys.exit(1)  # Exit the script with an error code

try:
    s_scale = int(sys.argv[1])
except ValueError:
    print("The s_scale argument must be an integer.")
    sys.exit(1)

output_dir=output_dir+str(s_scale)
BATCH_SIZE=int(512/s_scale)

#settings
UNK='[UNK]'
PAD='[PAD]'
max_vocab=20000 
WINDOW=5
negatives = 10
device = "cuda"
EPOCHS = 100
clip = 1 
char_embed_size=50 
gru_size=50*s_scale
gru_layers=s_scale
lin_layers=s_scale

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"The directory {output_dir} has been created.")
else:
    print(f"The directory {output_dir} already exists.")

model_filename="model.pt"


print('number of character %i' %(input_dim))
encoder_input_dim = input_dim  

print('reading input file: ', all_words_file)
with open(all_words_file,"rb") as f:
    all_words = pickle.load(f)

print('reading input file: ', words_file)
with open(words_file,"rb") as f:
    words = pickle.load(f)


print('reading input file: ', wv_file)
with open(wv_file,"rb") as f:
    wv = pickle.load(f)
    
max_word_len=0
def create_word2vec_and_vec2word(sentences_list, features_list):
    if len(sentences_list) != len(features_list):
        raise ValueError("The number of sentences and features lists should be the same.")
    word2vec_dict = {UNK:[int(1)], PAD:[int(0)]}
    vec2word_dict = {}
    global max_word_len
    for sentence, features in zip(sentences_list, features_list):
        if len(sentence) != len(features):
            raise ValueError("The number of words and features in each sentence should be the same.")

        for word, feature in zip(sentence, features):
            # Convert the features list to a tuple to make it hashable and usable as a dictionary key.
            if len(feature) > max_word_len:
                max_word_len=len(feature)
                if (max_word_len > 17):
                    print(word)
            feature_tuple = torch.FloatTensor(feature) + 2
            #feature_tuple = feature + 2
            # Create word2vec dictionary mapping words to features.
            word2vec_dict[word] = feature_tuple.tolist() #adding one to the features here
    print("max word length: ", max_word_len)
    for word in word2vec_dict.keys():
        a=word2vec_dict[word]
        a.extend([0 for _ in range(max_word_len-len(a))])
        word2vec_dict[word] = a
    return word2vec_dict, None

def build_vocab(words,wv, UNK='[UNK]', max_vocab=20000):
    sentences=words
    print("building vocab...")
    word2idx,idx2word = create_word2vec_and_vec2word(words,wv)
    vocab = set([word for word in word2idx])
    print("build done")
    return vocab, word2idx, idx2word

def word_to_idx(word, word2idx, UNK='[UNK]'):
    #global max_word_len
    if word in word2idx.keys():
        return word2idx[word]
    else:
        print(word)
        return word2idx[UNK]#_a

def char_pad_collate(batch):
  xx = batch
  xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
  return xx_pad

def process_line(line, words, UNK):
  for i in range(len(line)):
    if line[i] not in words:
      line[i]=UNK
  return line

def custom_collate_eval(batch):
    # Convert batch to a sequence of tensors
    #print(batch)
    batch_tensors = [torch.tensor(entry).int() + 2 for entry in batch]
    unk_tensor= word2chars[UNK]
    batch_tensors = [torch.LongTensor(unk_tensor)] + batch_tensors
    #print(batch_tensors)
    padded_batch = pad_sequence(batch_tensors, batch_first=True, padding_value=0)
    return padded_batch

def export_embeddings(words, embeddings, file_name):
    print('Writing embeddings to %s ...' % file_name)
    with io.open(file_name, 'w', encoding='utf-8') as f:
        f.write(u"%i %i\n" % (len(words),len(embeddings[0])))
        for i in range(len(words)):
            f.write(u"%s %s\n" % (words[i], " ".join('%.5f' % x for x in embeddings[i])))
            
#calculates pair
def do_eval(ogwords,ogwv):   
     
    # Initialize the defaultdict with list type
    batched_wv = defaultdict(list)
    for word, vec in tqdm(zip(ogwords, ogwv), total=len(ogwords)):
      word = word.split("_")[0].lower()
      if word in flattened_test_set:
        batched_wv[word].append(vec)

    sgns_w_encoder.eval()
    final_char_vectors = list()
    leftout = list()
    updated_words=list()
    for x in tqdm(list(batched_wv.keys())):
        tgs = custom_collate_eval(batched_wv[x]).to(device)
        if tgs.shape[0]>256:
            leftout.append(x)
            #continue
            tgs=tgs[0:256]
        #print(tgs)
        src = tgs.unsqueeze(1)          
        h = sgns_w_encoder(src).squeeze(1)
        res=h.detach().cpu().unsqueeze(0).numpy()
        final_char_vectors.append(res[0][1:])
        updated_words.append(x)
    final_char_vectors = [np.mean(x,axis=0) for x in final_char_vectors]
    export_embeddings(updated_words, final_char_vectors, os.path.join(output_dir,"temp.vec"))
    s_model = KeyedVectors.load_word2vec_format(os.path.join(output_dir,'temp.vec'))
    X=calc_test_dist(s_model, t_model.wv, test_set)
    Y=calc_test_dist(t_model.wv, s_model,  test_set)
    Z=cal_test_edit(t_model.wv, s_model, test_set)
    correlation, p_value = stats.pearsonr(X, Y)
    print(" w. target model cosine dist:", correlation)
    correlation, p_value = stats.pearsonr(X, Z)
    print(" w. edit distance:", correlation)
    sgns_w_encoder.train()

class CharEncoder(nn.Module):
    def __init__(self, input_dim, emb_size=50, hid_dim=50, n_layers=2, gru_layers=1, device='cuda', pretrained=None):
        super().__init__()
        self.device = torch.device(device)
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.gru_layers=gru_layers
        self.emb_size=emb_size
        self.embed = nn.Embedding(input_dim, emb_size)
        if pretrained is not None:
          #load from pretrained
          self.embed=nn.Embedding.from_pretrained(pretrained)
          self.embed.weight.requires_grad = False

        self.rnn = nn.GRU(emb_size, hid_dim, num_layers=gru_layers, bidirectional=True, batch_first = True)
        self.f=nn.ModuleList()
        for i in range(self.n_layers):
          self.f.append(nn.Linear(hid_dim*2, hid_dim*2).to(device))

    def forward(self, src):
        emb=self.embed(src)    
        emb=emb.view(-1, emb.shape[2], emb.shape[3])
        padding_mask = (src == 0).view(-1, emb.shape[1])
        emb[padding_mask] = 0
        non_zero_elements = emb.ne(0).any(dim=2)
        src_lengths = non_zero_elements.sum(dim=1)
        src_lengths, perm_idx = src_lengths.sort(0, descending=True)
        emb = emb[perm_idx]
        packed_src = pack_padded_sequence(emb, src_lengths.cpu(), batch_first=True, enforce_sorted=True)
        packed_outputs, hidden = self.rnn(packed_src)
        _, unperm_idx = perm_idx.sort(0)
        hidden = hidden[:, unperm_idx]
        hidden=hidden[self.gru_layers*2-2:] 
        hidden=hidden.view(hidden.shape[0], src.shape[0], -1, self.hid_dim)
        hidden_enc = torch.cat((hidden[0],hidden[1]), dim=2)

        for i in range(self.n_layers):
          hidden_enc = self.f[i](hidden_enc)
        return hidden_enc



def pad_collate(batch):
  feats = [batch[i][0] for i in range(len(batch))] 
  targs = [batch[i][1].tolist() for i in range(len(batch))] 
  #print(targs)
  padded_feats = pad_sequence(feats, batch_first=True, padding_value=0)  #then set ignore index = 0
  return padded_feats, torch.LongTensor(targs)


print('\nProcessing text file:', text_filename)
#process dataset
file = open(text_filename, 'rt')
text = file.readlines()
file.close()

n_s=0
n_w=0
s_sentences=[]
word_idx=0
for sent in text:
  new_sen=[]
  tokens=word_tokenize(sent)
  for word in tokens:#sent.split():
    new_sen.append(word.lower())
    n_w+=1
  s_sentences.append(new_sen)
  n_s+=1

print("Number of sentences:", n_s)
print("Number of words:", n_w)

print("training target (gensim) model")
t_model=Word2Vec(s_sentences)
c_model = KeyedVectors.load_word2vec_format('char_embeddings.vec') #TODO
test_set=create_test_set(s_sentences, t_model.wv, c_model)
flattened_test_set = list(set(list(flatten(test_set))))


ctr = 0
nwv,nwords = list(),list()
currwv,currwords=list(),list()
for i in tqdm(range(len(all_words))):
    if all_words[i]=="\n":
        nwv.append(currwv)
        nwords.append(currwords)
        currwv,currwords=list(),list()
    else:
        new_vec=wv[ctr]
        currwv.append(new_vec)
       # currwv.append(wv[ctr])
        currwords.append(words[ctr])
        ctr+=1
print("nwords here : ",nwords[0])
nwords = add_suffix_to_lists_of_words(nwords)
print("nwords next : ",nwords[0])
ogwv=wv
wv=nwv#[0:1000]
ogwords=words
words=nwords#[0:1000]
del nwords,nwv

vocab, word2idx, _ = build_vocab(words,wv, UNK=UNK, max_vocab=max_vocab)


#Below, load examples. Covert positive examples to sequences of characters. Convert targets to list
train_dataset = datasets.WordsCBOWDataset(words, word2idx, WINDOW)
data_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False)

print('\n\nConstructing Character-Based Corpus - Sequences of Context Words, each is a padded sequence of chars')
_features = []
_targets =[]


word2chars=word2idx
word2chars[UNK]=[1] + torch.LongTensor(np.zeros(max_word_len-1)).tolist()

for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
    words=batch[:][0].tolist()
    _temp=[]
    for _word in words:
      _temp.append(_word)
    _features.append(torch.LongTensor(_temp[1:]))
    _targets.append(torch.LongTensor(_temp[0]))

print(_features[0])
print(_targets[0])

#updated classes

audio_train_data=AudioDataset(_features, _targets)

data_loader = DataLoader(dataset=audio_train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate)

encoder_input_dim = input_dim  

print("\n\nBuilding SGNS model with integrated char encoder")

sgns_w_encoder = CharEncoder(encoder_input_dim,  emb_size=char_embed_size, hid_dim=gru_size, n_layers=s_scale, gru_layers=gru_layers, device=device).to(device)
sgns_c_encoder = CharEncoder(encoder_input_dim,  emb_size=char_embed_size, hid_dim=gru_size, n_layers=s_scale, gru_layers=gru_layers, device=device).to(device)

#load existing model #TODO
if resume_training:
  try:
    checkpoint_model = os.path.join(output_dir, model_filename)
    loaded_pt = torch.load(checkpoint_model)
    sgns_w_encoder.load_state_dict(loaded_pt['model_state_dict'])
    sgns_c_encoder.load_state_dict(loaded_pt['model_state_dict_c'])
    params = list(sgns_w_encoder.parameters()) + list(sgns_c_encoder.parameters())
    optimizer = optim.AdamW(params) 
    if s_scale > 2:
          optimizer = optim.AdamW(params,lr=1e-4)

    optimizer.load_state_dict(loaded_pt['optimizer_state_dict'])
    best_loss=loaded_pt['loss']
    print('resuming training from ', checkpoint_model)
    last_epo = loaded_pt['epoch'] + 1

  except Exception :
    last_epo=0
    print("Couldn't resume training, loading fresh model\n")
    params = list(sgns_w_encoder.parameters()) + list(sgns_c_encoder.parameters())
    optimizer = optim.AdamW(params)#, lr=1e-4, eps = 1e-6)
    if s_scale > 2 :
      optimizer = optim.AdamW(params, lr=1e-4)#, eps = 1e-6)


print(sgns_w_encoder)
print(sgns_c_encoder)

encoder_total_params = sum(p.numel() for p in sgns_w_encoder.parameters())
print("Word encoder parameters: ", encoder_total_params)


cc_loss=nn.CrossEntropyLoss()
sgns_w_encoder.train()
sgns_c_encoder.train() 

#Negative samples pool
print("Constructing negative examples pool ...")
all_word_chars=[]
for k, v in word2chars.items():
  temp=v#.tolist()
  temp.extend(0 for _ in range(max_word_len-len(v)))
  all_word_chars.append(temp[:max_word_len])

all_word_chars=torch.LongTensor(all_word_chars[2:])
print("done. ")
#create dataloader for saving word embeddings:
def char_pad_collate(batch):
  xx = batch
  xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
  return xx_pad


"""#Train model"""
print("Training model:")
sgns_w_encoder.train()
sgns_c_encoder.train()
losses = []
best_loss = math.inf
do_eval(ogwords, ogwv)
for epo in range(EPOCHS-last_epo):
  epoch_loss = 0
  for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
      tgs = batch[1].to(device)
      pos=batch[0].to(device) 

      samples = FT(pos.shape[0]*WINDOW * negatives).uniform_(0, len(all_word_chars) - 1).long()
      neg = all_word_chars[samples]
      neg= neg.view(pos.shape[0], WINDOW * negatives, -1).to(device)

      tgs=sgns_w_encoder(tgs.unsqueeze(1)).squeeze(1).unsqueeze(2)
      pos=sgns_c_encoder(pos)
      neg=sgns_c_encoder(neg).neg()
      optimizer.zero_grad()

      oloss = torch.bmm(pos, tgs).squeeze().sigmoid().log().mean(1)
      nloss = torch.bmm(neg, tgs).squeeze().sigmoid().log().view(-1, WINDOW, negatives).sum(2).mean(1)
      loss= -(oloss + nloss).mean()

      if torch.isfinite(loss):
        loss.backward()           
        optimizer.step()
        epoch_loss += loss.item()
      else:
        print("infinite loss detected")

  epoch_loss /= len(data_loader)
  losses.append(epoch_loss)
  print("Epoch:{}, train loss:{} ".format(epo + 1 + last_epo,epoch_loss))
  do_eval(ogwords, ogwv)
  if epoch_loss<best_loss:
    best_loss=epoch_loss
    #save model
    print("saving model ... ")
    torch.save({'epoch': epo + last_epo,
            'model_state_dict': sgns_w_encoder.state_dict(),
            'model_state_dict_c': sgns_c_encoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_loss,
            }, os.path.join(output_dir, model_filename))




