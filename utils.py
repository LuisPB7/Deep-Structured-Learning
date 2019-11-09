import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from variables import *
import unicodedata
import pickle
import csv
import numpy
import copy
import json
import re
import spacy

torch.manual_seed(0)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

# Fetch word2id dictionary #
with open("vocabulary_dict.pkl", 'rb') as handle:
    word2idx = pickle.load(handle)

# Fetch embedding weights #
with open("embedding_weights.pkl", 'rb') as handle:
    embedding_weights = torch.tensor(pickle.load(handle))
    
def create_emb_layer(non_trainable=False):
    weights_matrix = embedding_weights
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False
    return emb_layer

# Class to pad variable length sequence according to biggest sequence in the batch
class PadSequence:
    
    def __call__(self, batch):    
        premises = []
        hyps = []
        labels = []
        for x in batch:
            premises.append(x[0])
            hyps.append(x[1])
            labels.append(x[2])
            
        # Pad premises
        premises_padded = torch.nn.utils.rnn.pad_sequence(premises, batch_first=True)
        lengths_premises = torch.LongTensor([len(x) for x in premises])
        
        # Pad hypothesis
        hyps_padded = torch.nn.utils.rnn.pad_sequence(hyps, batch_first=True)
        lengths_hyps = torch.LongTensor([len(x) for x in hyps])
        
        return premises_padded, lengths_premises, hyps_padded, lengths_hyps, torch.tensor(labels)

class NLIDataset(Dataset):
  
    def __init__(self, name):
        super(NLIDataset, self).__init__()
        
        self.data = []
        with open(name, 'r') as json_file:
            json_list = list(json_file)
            for json_str in json_list:
                instance = json.loads(json_str)
                if instance['gold_label'] not in ['neutral', 'entailment', 'contradiction']:
                    continue
                self.data.append(instance)

        self.word2idx = word2idx
        self.nlp = spacy.load('en')
    
    def sentenceToIndex(self, sentence):
        sentence = [token.text.lower() for token in self.nlp(sentence)] # Spacy is responsible for tokenization
        idx = []
        for token in sentence:
            try:
                idx += [self.word2idx[token]]
            except:
                idx += [UNK_TOKEN] #UNK token
        return torch.tensor(idx, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        premise = self.sentenceToIndex(self.data[index]['sentence1'])
        hypothesis = self.sentenceToIndex(self.data[index]['sentence2'])
        label = self.data[index]['gold_label']
        
        # [NEUTRAL, ENTAILMENT, CONTRADICTION]
        if label == 'neutral':
            label = torch.tensor(0)
        elif label == 'entailment':
            label = torch.tensor(1)
        else:
            label = torch.tensor(2)
        return premise, hypothesis, label
