# Spoken Word2Vec

<a href='https://arxiv.org/pdf/2311.09319'><img src='https://img.shields.io/badge/paper-Paper-red'></a> 

This repo contains python implementation of spoken word2vec models described in the following paper:

```
@inproceedings{spokenW2V,
  author={Mohammad Amaan Sayeed and Hanan Aldarmaki},
  title={{Spoken Word2Vec: Learning Skipgram Embeddings from Speech}},
  year=2024,
  booktitle={Proceedings of INTERSPEECH 2024}
}
```

These scripts are extensions of the character-based skipgram models [available here](https://github.com/h-aldarmaki/Word2Vec/tree/main). 


## Sample Data

We provide a subset of files from LibriSpeech dev-clean to illustrate the expected directory structure for the feature extraction scripts. To replicate the performance in the paper, you need to generate features for the whole [LibriSpeech ASR Corpurs train-clean-100 set](https://www.openslr.org/12). 

[Alignment files](https://github.com/CorentinJ/librispeech-alignments) are needed for identifying word boundaries

## Dependencies

The scripts run in python 3.x. You will need the following packages:

```
os, tqdm, argparse, pickle, pandas
torch, numpy, s3prl, sklearn, librosa
nltk, Levenshtein, gensim

```
We tested the code on the following versions: python 3.11.7, torch 1.13.1, scikit-learn 1.2.1

You will also need sufficient storage and system RAM for feature extraction. For example, if you extract HuBERT features, you will need at least **86GB** of storage to run steps 1 and 2 for the train-clean-100 set, and more than 100GB of system RAM. In our experiments, we ran the code using one A100 GPU (40GB GPU RAM) and 230GB system RAM. 

## Steps

### Feature Extraction

```
python step_1_extract_features.py --feature_type hubert
```
