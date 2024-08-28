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

We provide a subset of files from LibriSpeech dev-clean to illustrate the expected directory structure for the feature extraction scripts. To replicate the performance in the paper, you need to generate features for the whole [LibriSpeech ASR Corpurs train-clean-100 set](https://www.openslr.org/12). You will need **32G** of storage for the features generated in steps 1 and 2 for the train-clean-100 set. 

[LibriSpeech Corpus](https://www.openslr.org/12)

[Alignment files](https://github.com/CorentinJ/librispeech-alignments)

## Dependencies

The scripts run in python 3.x. You will need the following packages:

```
os, tqdm, argparse, pickle, pandas
torch, numpy, s3prl, sklearn, librosa
nltk, Levenshtein, gensim

```

## Training The Model

[TODO]
