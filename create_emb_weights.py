import numpy as np
import pickle
import operator
import jellyfish
from variables import *
import spacy
import json

nlp = spacy.load('en')

# Get dictionary with word frequencies in SNLI train dataset #
with open('snli_1.0_train.jsonl', 'r') as json_file:
    json_list = list(json_file)
    word_dict = {}
    for json_str in json_list:
        instance = json.loads(json_str)
        premise = [token.text.lower() for token in nlp(instance['sentence1'])] 
        hyp = [token.text.lower() for token in nlp(instance['sentence2'])] 
        for token in premise+hyp:
            try:
                word_dict[token] += 1
            except:
                word_dict[token] = 0

# Sort tokens by their frequency (we only want VOCAB_SIZE most frequent) #
sorted_freqs = sorted(word_dict.items(), key=operator.itemgetter(1), reverse=True)

vocabulary = sorted_freqs[0:VOCAB_SIZE-2]
vocabulary_dict = {} 
for token, n in vocabulary:
    vocabulary_dict[token] = len(vocabulary_dict)+1 # vocabulary_dict maps token->integer id

pickle.dump(vocabulary_dict, open("vocabulary_dict.pkl", 'wb'))

glove = {}
with open('glove.42B.300d.txt', 'rb') as f:
    for l in f:
        line = l.decode().split()
        word = line[0]
        vect = np.array(line[1:]).astype(np.float)
        glove[word] = vect

matrix_len = VOCAB_SIZE
weights_matrix = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))
UNK_index = VOCAB_SIZE-1

def closest_word(word):
    # check closest GLOVE word
    return max(glove, key=lambda x:jellyfish.jaro_winkler(x, word))

# Now we create the embedding matrix for the the neural network #
for word, i in vocabulary_dict.items():
    try: 
        weights_matrix[i] = glove[word]
    except KeyError:
        print("Closest word to {} is {}".format(word, closest_word(word)))
        weights_matrix[i] = glove[closest_word(word)] # No GLOVE vector? Get GLOVE vector of closest word (there might be a mis-spelling...)
        
# UNK token
weights_matrix[UNK_index] = np.random.normal(scale=0.6, size=(EMBEDDING_DIM, ))
pickle.dump(weights_matrix, open('embedding_vectors.pkl', 'wb'))
