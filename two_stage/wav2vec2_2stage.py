!pip install Levenshtein
import s3prl
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm.notebook  import tqdm
import random
import numpy as np
from random import randint
from torch.nn.utils.rnn import pad_sequence
from sklearn.cluster import MiniBatchKMeans as KMeans
from sklearn.metrics import adjusted_rand_score, homogeneity_score, completeness_score, v_measure_score
from collections import Counter
# from google.colab import drive
import math
# from Levenshtein import distance
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
from matplotlib import pyplot as plt
from tqdm import tqdm
import os,pickle,numpy
from tqdm import tqdm
from sklearn import preprocessing
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import os,pickle,numpy
from tqdm import tqdm

#Import statements
import string

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec, KeyedVectors
from pandas.core.common import flatten
import pandas as pd
import random
from tqdm import tqdm
import os
import numpy as np
import scipy
import sys
import gensim.downloader as gensim_api
from statistics import mean, variance
from Levenshtein import distance as edit_distance
from scipy import stats
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
# drive.mount('/content/gdrive')

def length_to_mask(length, max_len=None, dtype=None):
    """length: B.
    return B x max_len.
    If max_len is None, then max of length will be used.
    """
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or length.max().item()
    mask = torch.arange(max_len, device=length.device,
                        dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
    return mask


class Encoder(nn.Module):
    def __init__(self, input_dim,  hid_dim=100, n_layers=2, dropout=0.3):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.hid_dim = hid_dim
        self.n_layers = n_layers   
        self.device = torch.device("cuda")    
        self.rnn = nn.GRU(input_dim, hid_dim, n_layers, bidirectional=True, batch_first=True)

#         self.rnn = nn.LSTM(input_dim, hid_dim, n_layers, bidirectional=True, batch_first = True)
        
    def forward(self, src, src_len):
        #packed_src = nn.utils.rnn.pack_padded_sequence(src, src_len.to('cpu'), enforce_sorted=True, batch_first=True )
        packed_src = nn.utils.rnn.pack_padded_sequence(self.dropout(src), src_len.to('cpu'), enforce_sorted=True, batch_first=True )
        outputs, hidden= self.rnn(packed_src, None)
#         outputs, (hidden,cell)= self.rnn(packed_src, None)
        return hidden[self.n_layers*2-2:] 

class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim=100, n_layers=2):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.input_dim = output_dim+hid_dim*2
        #self.input_dim = 50+output_dim+hid_dim*2
        #self.pos_emb = nn.Embedding(max_len, 50)
        #self.max_len=max_len
        self.rnn = nn.GRU(self.input_dim, self.hid_dim, n_layers, bidirectional=False, batch_first=True)

#         self.rnn = nn.LSTM(self.input_dim, self.hid_dim, n_layers, bidirectional=False, batch_first = True)
        self.fc_out = nn.Linear(self.hid_dim, self.output_dim)
        
    def forward(self, input, hidden, hidden_enc):
        #emb=self.pos_emb(_input) .unsqueeze(1)
        #rnn_input=torch.cat((emb, input, hidden_enc), dim=2)
        rnn_input=torch.cat((input, hidden_enc), dim=2)
        output, hidden = self.rnn(rnn_input, hidden)     
        prediction = self.fc_out(output.squeeze(1))        
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()       
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, src_len, epoch=0):    
        batch_size = src.shape[0]
        decoder_hidden_size = self.decoder.hid_dim
        hidden_enc = self.encoder(src, src_len)
        hidden_enc = torch.cat((hidden_enc[0],hidden_enc[1]), dim=1).unsqueeze(1)
        trg_len=src.shape[1]  
        outputs = torch.zeros(batch_size, trg_len, self.decoder.output_dim).to(self.device)
        start_mfcc = torch.zeros( (src.shape[0], 1, src.shape[2]) , device=self.device) 
        #pos_input = torch.zeros(batch_size, dtype=torch.int32) 
        #output, hidden = self.decoder(pos_input.to(self.device),start_mfcc, None, hidden_enc)
        output, hidden = self.decoder(start_mfcc, None, hidden_enc)
        outputs[:,0,:] = output.squeeze(1)
        #if trg_len > self.decoder.max_len:
        #  trg_len = self.decoder.max_len
        for t in range(1, trg_len):
            prev_output = output.detach().unsqueeze(1)
        #    pos_input = torch.zeros(batch_size, dtype=torch.int32)*t
        #    output, hidden = self.decoder(pos_input.to(self.device), prev_output, hidden, hidden_enc) 
            output, hidden = self.decoder(prev_output, hidden, hidden_enc) 
            outputs[:,t,:] = output.squeeze(1)  
        
        mask = length_to_mask(torch.LongTensor(src_len).to(self.device))
        outputs = outputs * mask.unsqueeze(2)
        return outputs



wv = list()
words = list()
with open("/l/users/mohammad.sayeed/scp_files_train_clean_s3prl_wav2vec2/train_clean_100_s3prl.scp", 'r') as fin1:
    for line in tqdm(fin1):
        with open(os.path.join("/l/users/mohammad.sayeed/scp_files_train_clean_s3prl_wav2vec2/", line.split("\n")[0]), 'rb') as fin2:
            data = pickle.load(fin2)
            for feat in data:
                if feat[0]!='\n':
                    wv.append(torch.tensor(feat[0]))
                    words.append(feat[1])
                del feat
            del data

#Change the following as needed
#MFCC features:

nwords = list()
for idx,w in enumerate(words):
    nwords.append(w+"_"+str(idx))
words = nwords
del nwords
input_dim = 768
train_file='/content/gdrive/My Drive/miniASR_data/miniasr_features/dev-clean-mfcc.pt'
test_file='/content/gdrive/My Drive/miniASR_data/miniasr_features/test-clean-mfcc.pt'
model_save_dir = "/content/gdrive/My Drive/miniASR_data/models/audio2vec_model_mfcc_h250_1layer.pt"

#these are probably the same
words_file='/content/gdrive/My Drive/miniASR_data/miniasr_features/test-clean-words.pt'
speakers_file = '/content/gdrive/My Drive/miniASR_data/miniasr_features/test-clean-speakers.pt'


encoder_input_dim = input_dim  
decoder_hidden_dim = 100
encoder_hidden_dim = 100
decoder_output_dim = input_dim
BATCH_SIZE = 256
device = torch.device("cuda")


enc = Encoder(encoder_input_dim, encoder_hidden_dim,  n_layers=2)
dec = Decoder(encoder_input_dim,decoder_hidden_dim,  n_layers=2)
model = Seq2Seq(enc,dec,device).to(device)

optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=0.001)
# scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.3)

# optimizer = optim.SGD(model.parameters(), momentum = 0.9, lr = 0.001)
# scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.3)


mse_loss = nn.MSELoss()
best_loss = math.inf

print(model)


def pad_collate(batch):
  lengths = [len(x) for x in batch]
  padded_mfccs = pad_sequence(batch, batch_first=True, padding_value=0)

  return padded_mfccs, lengths

class AudioDataset(Dataset):
  def __init__(self, wv,sort=False):
        'Initialization'
        self.mfcc_features = wv#torch.load(file_mfcc)
        if sort:
          if self.mfcc_features[0].dim() < 3:
           self.mfcc_features = sorted(self.mfcc_features, key=lambda x: x.size()[0], reverse=True)
          else:
           self.mfcc_features = sorted(self.mfcc_features, key=lambda x: x.size()[1], reverse=True)
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.mfcc_features)
  def __getitem__(self, index):
        'Generates one sample of data'
        return self.mfcc_features[index].squeeze(0)
    
print("Loading data")
train_dataset = AudioDataset(wv)
print("training dataset size: "+str(len(train_dataset)))
train_loader = DataLoader(train_dataset, shuffle=False,  batch_size=BATCH_SIZE, collate_fn=pad_collate)
test_dataset = train_dataset #AudioDataset(wv[:-100])
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, collate_fn=pad_collate)
del wv
avg_embs=[]
untrained_embs=[]
model.eval()

for i, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
    src = batch[0]
    src_len = batch[1]
    _sum=torch.sum(src, dim=1)
    i=0
    for _e in _sum:
        _e=_e/src_len[i]
        avg_embs.append(_e.detach().cpu().numpy())
        i=i+1
    src=src.to(device).float()
    src_len, _indices = torch.sort(torch.LongTensor(src_len), dim=0, descending=True, stable=True)
    src=src[_indices]
    h = model.encoder(src, torch.LongTensor(src_len).float())
    _,_orig=torch.sort(_indices, dim=0, stable=True)
    hidden = torch.cat((h[0],h[1]), dim=1)                
    hidden = hidden.detach().cpu().numpy()
    hidden=hidden[_orig,:]
    for emb in hidden:
        untrained_embs.append(emb)

print("untrained embeddings size:",len(untrained_embs[0]))
print("average embeddings size:",len(avg_embs[0]))
model_save_dir = "./audioEncoder_20_epochs_s3prl_trainclean_wav2vec.pt"
EPOCHS = 200 
losses = []

for epo in range(EPOCHS):
  epoch_loss = 0
  model.train()
  for batch in tqdm(train_loader, total=len(train_loader)):
      src = batch[0].to(device)
      src_len = batch[1]
      src_len, _indices = torch.sort(torch.LongTensor(src_len), dim=0, descending=True, stable=True)
      src=src[_indices]
      optimizer.zero_grad()
      output = model(src, torch.LongTensor(src_len), epo)
      output_dim = output.shape[-1]
      output = output[0:].view(-1, output_dim)
      src = src[0:].view(-1, src.shape[-1])     
      loss = mse_loss(output.float(),src.float())    
      loss.backward()
      optimizer.step()
#       scheduler.step()
      epoch_loss += loss.item()

  epoch_loss /= len(train_loader)
  print("Epoch:{}, train loss:{} ".format(epo + 1,epoch_loss))
  if epoch_loss < best_loss:
    best_loss=epoch_loss
    print("saving model in", model_save_dir)
    torch.save(model.state_dict(), model_save_dir)


model.load_state_dict(torch.load("./audioEncoder_20_epochs_s3prl.pt"))
model.load_state_dict(torch.load(model_save_dir))
# model.load_state_dict(torch.load("./audioEncoder_20_epochs_s3prl.pt"))
trained_embs=[]
model.eval()

for i, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
                src = batch[0].to(device)
                src_len = batch[1]
                src_len, _indices = torch.sort(torch.LongTensor(src_len), dim=0, descending=True, stable=True)
                src=src[_indices]
                h = model.encoder(src, torch.LongTensor(src_len))
                _,_orig=torch.sort(_indices, dim=0, stable=True)
                hidden = torch.cat((h[0],h[1]), dim=1)             
                hidden = hidden.detach().cpu().numpy()
                hidden=hidden[_orig,:]
                for emb in hidden:
                    trained_embs.append(emb)

print("No.of embeddings obtained -",len(trained_embs))

words_occ = words
words = [w.split("_")[0] for w in words]
words = [x.split("_")[0] for x in words_occ]
#~10 mins to run:
K=len(set(words))
print('k-means clustering with k =', K)
kmeans = KMeans(n_clusters=K, init='random').fit(trained_embs)  
 
h_score = homogeneity_score(words, kmeans.labels_)
c_score = completeness_score(words, kmeans.labels_)
v_score = v_measure_score(words, kmeans.labels_)

print('homogenity: ', h_score)
print('completeness: ', c_score)
print('v score: ', v_score)


def avg_editDist(word_list):
    transformed_strings = np.array(word_list).reshape(-1,1)
    A=pdist(transformed_strings,lambda x,y: distance(x[0],y[0])/max(len(x[0]), len(y[0])))
    A=squareform(A)
    #sum the off diagonal elements, and divide by 2 (symmetric)
    sum = (A.sum() - np.diag(A).sum())/2 
    n=A.shape[0]
    total = n*(n-1)/2
    return sum/total

_words_=np.array(words)

sum=0
total=0
list_dist=[]
for cid in range(K):
  idx=np.where(kmeans.labels_ == cid)[0]
  _words=_words_[idx]
  if(len(_words) > 1):
   _dist=avg_editDist(_words)
   _len=len(_words)
   list_dist.append(_dist)
  elif len(_words) == 1:
   _dist=0
   _len=1
   list_dist.append(-0.1)
  else:
   _dist=0
   _len=0
   list_dist.append(-0.1)

  sum+=_dist*_len
  total+=_len

print("\naverage edit distance within each cluster:" , sum/total)

#calculate labeling based on clusters:
_labels_=[]
for cid in range(K):
  idx=np.where(kmeans.labels_ == cid)[0]
  _words=_words_[idx]
  if(len(_words) > 1):
    _c=Counter(_words)
    _w=_c.most_common(1)[0][0]
    _labels_.append(_w)
  elif len(_words) == 1:
    _labels_.append(_words[0])
  else:
    _labels_.append("")


correct=0
edit_dist_total=0
incorrect=0
for i in range(len(words)):
  if _labels_[kmeans.labels_[i]] == words[i]:
    correct+=1
  else:
    edit_dist_total+=distance(_labels_[kmeans.labels_[i]], words[i])/max(len(_labels_[kmeans.labels_[i]]), len(words[i]))
    incorrect+=1
print('\naccuracy of cluster labeling: ', correct/len(words))
print('avg edit distance for incorrect labels ', edit_dist_total/incorrect)


_words_=np.array(words)

print('\n\nFirst 10 clusters')
for cid in range(10):
  idx=np.where(kmeans.labels_ == cid)[0]
  print(_words_[idx])
df = pd.DataFrame()
df['word'] = list(words_occ)
df['emb'] = list(trained_embs)
labels, counts = np.unique(kmeans.labels_, return_counts=True)
print(dict(zip(labels, counts)))
# grouped_df = df.groupby("clusters").agg({"word":list,'emb':list})
# grouped_df['max_word'] = grouped_df['word'].apply(lambda x : [y.split("_")[0] for y in x]).apply(lambda lst : max(set(lst), key=lst.count))
# grouped_df.head(5)
df['clusters'] = kmeans.labels_
df['index_to_sort'] = [i for i in range(len(df))]
df = df.sort_values(by='index_to_sort')
clusters_corresponding_max_word = list()
for x in tqdm(df.clusters):
    clusters_corresponding_max_word.append(_labels_[x])
df['max_word'] = clusters_corresponding_max_word
df.head(2)
matches=0
for idx in tqdm(range(len(df))):
    if df.iloc[idx]['word'].split("_")[0]==df.iloc[idx]['max_word']:
        matches+=1
print(matches/len(df))

all_words = list()
with open("/l/users/mohammad.sayeed/scp_files_train_clean_s3prl_wav2vec2/train_clean_100_s3prl.scp", 'r') as fin1:
    for line in tqdm(fin1):
        with open(os.path.join("/l/users/mohammad.sayeed/scp_files_train_clean_s3prl_wav2vec2/", line.split("\n")[0]), 'rb') as fin2:
            data = pickle.load(fin2)
            for feat in data:
                all_words.append(feat[1])
                del feat
            del data

ctr = 0
s_sentences=list()
t_sentences=list()
for i in tqdm(range(len(all_words))):
    if all_words[i]=="\n":
        s_sentences.append("\n")
        t_sentences.append("\n")
    else:
        s_sentences.append(df.iloc[ctr]['clusters'])
        t_sentences.append(df.iloc[ctr]['word'])
        ctr+=1
        

#Noisy datasets
def addNoise(sentences, df, rate):
  res=[]
  samples=df.sample(n=len(sentences)*100, replace=True, weights='counts').index.values
  idx=0
  for sent in tqdm(sentences, total=len(sentences)):
    _s = []
    for word in sent:
      if ( random.random() < rate): #e.g. 10% of the time
         _s.append(samples[idx])
         idx+=1
      else:
        _s.append(word)
    res.append(_s)
  return res


#Procrustes Alignment
def align(s_vecs, t_vecs, seed_words):
  A=[]
  B=[]
  for word in seed_words:
      s_str=word
      t_str=word
      if test_word_2_cluster(s_str) in s_vecs.wv.index_to_key and t_str in t_vecs.wv.index_to_key :
        A.append(s_vecs.wv[test_word_2_cluster(s_str)])
        B.append(t_vecs.wv[t_str])

  A=np.asarray(A, dtype=np.float32)
  B=np.asarray(B, dtype=np.float32)
  A=np.transpose(A)
  B=np.transpose(B)

  M=np.matmul(B,np.transpose(A))
  U,S,V_t = np.linalg.svd(M, full_matrices=True)
  R=np.matmul(U, V_t)

  return R

#returns average semantic sim, and everage edit dist (avergaed for 5 nearest neighbors, then averaged for all test words)
def evaluate_embeddings(words, s_model, t_model):
  #find 5 nearest neighbors for each word, calculate sim via t_model, average across words, and across all
  all_sim=[]
  all_edit=[]
  total=0
  for word in words:
    nns = [w[0] for w in s_model.wv.most_similar(word, topn=5)]
    sims = [t_model.wv.similarity(word, w) for w in nns if w in t_model.wv.index_to_key]
    edits = [edit_distance(word, w)/max(len(w), len(word)) for w in nns if w in t_model.wv.index_to_key]
    if len(sims) > 0:
      all_sim.extend(sims)
      all_edit.extend(edits)
  print('average semantic sim', np.mean(all_sim), ', variance: ', np.var(all_sim))
  print('average edit distance', np.mean(all_edit), ', variance: ', np.var(all_edit))
  return all_sim, all_edit
    

def create_test_set(sentences, s_model, c_model):
    #create FreqDist object
    unigram_dist = nltk.FreqDist(flatten(sentences))
    #first, convert the unigram_dist object to a pd.DataFrame object:
    unigrams_df = pd.DataFrame(pd.Series(unigram_dist))
    #rename the columns 
    unigrams_df.columns = ['counts']
    #Sort the values by count
    unigrams_df = unigrams_df.sort_values('counts', ascending=False)
    temp={w[0] for w in unigram_dist.most_common(5000)}
    temp=list(temp)[200:]
    random.shuffle(temp)
    _test_words=temp[1:2000]
    test_pool=temp[2000:]

    test_set = []
    for word in _test_words:
        _t=[word]
        pool=s_model.most_similar(word, topn=5)
        _t.extend([w[0] for w in pool])
        pool=c_model.most_similar(word, topn=5)
        _t.extend([w[0] for w in pool])
        _t.extend(random.sample(test_pool, 10))
        if len(_t) > 1:
            test_set.append(_t)

    return test_set

def calc_test_sim(vecs1, vecs2, testset):
  res=[]
  for item in testset:
    s=item[0]
    #TODO change vocab to key_to_index
    if s in vecs1.vocab and s in vecs2.vocab:
       sims = [vecs1.similarity(s, w) for w in item[1:] if w in vecs1.vocab and w in vecs2.vocab]
       res.extend(sims)
  return res

def cal_test_edit(vecs1, vecs2, testset):
  res=[]
  for item in testset:
    s=item[0]
    #TODO change vocab to key_to_index
    if s in vecs1.vocab and s in vecs2.vocab:
       edits = [edit_distance(s, w)/max(len(w), len(s)) for w in item[1:] if w in vecs1.vocab and w in vecs2.vocab]
       res.extend(edits)
  return res

text = " ".join([str(x) for x in s_sentences]).split("\n")
n_s=0
n_w=0
s_sentences=[]
for sent in text:
  new_sen=[]
  tokens=word_tokenize(sent)
  for word in tokens:#sent.split():
   # if word not in string.punctuation:
            new_sen.append(word.lower())
            n_w+=1
  s_sentences.append(new_sen)
  n_s+=1

print("Number of sentences:", n_s)
print("Number of words:", n_w)
print(" ".join(s_sentences[0]))
print(" ".join(s_sentences[1]))
print(" ".join(s_sentences[3]))


transformed = []

def process_cluster(x):
    return list(df.loc[df['clusters'] == int(x)]['max_word'])[0].lower()

for sentence in tqdm(flatten_sentences):
    transformed.append(process_cluster(sentence))

#Create dataframes for sampling

#create FreqDist object
unigram_dist = nltk.FreqDist([x.lower() for x in list(set(transformed))])
#first, convert the unigram_dist object to a pd.DataFrame object:
unigrams_df = pd.DataFrame(pd.Series(unigram_dist))
#rename the columns 
unigrams_df.columns = ['counts']
#Sort the values by count
unigrams_df = unigrams_df.sort_values('counts', ascending=False)


#Extract dictionary
t_unigram_dist = nltk.FreqDist(flatten(t_sentences))

temp1={w[0] for w in unigram_dist.most_common(5000)}
temp2={w[0] for w in t_unigram_dist.most_common(5000)}
temp=temp1 & temp2
temp=list(temp)
print(len(temp))
random.shuffle(temp)
seed_words=temp[:len(temp)-int(len(temp)-(len(temp)*0.8))]
test_words=temp[len(temp)-int(len(temp)-(len(temp)*0.8)):]
t_model = Word2Vec(t_sentences, vector_size=100, min_count=1, epochs=50, seed=random.randint(0, 10000))
s_model = Word2Vec(s_sentences, vector_size=100, min_count=1, epochs=50, seed=random.randint(0, 10000))
s_model_wv_index_to_key = list()
s_model_wv = list()
for x in tqdm(s_model.wv.index_to_key):
    l = list(df.loc[df['clusters'] == int(x)]['max_word'])
    s_model_wv_index_to_key.append(str(max(l,key=lambda x : l.count(x))).lower())
    s_model_wv.append(s_model.wv[x])


#calculate pairwise editDist. Calculate pairwise cosine. Calculate pearson corr. 

# test_words = list(set(t_model.wv.index_to_key).intersection(set(s_model_wv_index_to_key)))

ntest_words=list()
for w in test_words:
    if w in s_model_wv_index_to_key and w in t_model.wv.index_to_key:
        ntest_words.append(w)
print(len(ntest_words))
test_words = ntest_words
del ntest_words

def test_word_2_cluster(x):
    try:
        return s_model.wv.index_to_key[s_model_wv_index_to_key.index(x)]
    except Exception:
        return False
def p_editDist(word_list):
    transformed_strings = np.array(word_list).reshape(-1,1)
    A=pdist(transformed_strings,lambda x,y: edit_distance(x[0],y[0])/max(len(x[0]), len(y[0])))
    return A

def p_cosine(A):
    scores = pdist(A, metric='cosine')
    return scores

np.random.seed(1032)

#sub_idx=np.random.choice(np.array(range(0, len(final_words))), size=1000, replace=False)

test_embs_s=np.array(s_model.wv[[test_word_2_cluster(w) for w in test_words]])
X=p_editDist(test_words)
Y=p_cosine(test_embs_s)
correlation, p_value = stats.pearsonr(X, Y)
print("Pearson correlation w. edit distance:", correlation)

test_embs_t=np.array(t_model.wv[test_words])
X=p_cosine(test_embs_s)
Y=p_cosine(test_embs_t)
correlation, p_value = stats.pearsonr(X, Y)
print("Pearson correlation w. target sim:", correlation)
