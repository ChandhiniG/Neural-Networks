import pickle
import nltk
from pycocotools.coco import COCO
from collections import defaultdict
import csv
import numpy as np
import torch

def get_glove(word2ind, path, embedding_dim=50):
    with open(path) as f:
        embeddings = np.zeros((len(word2ind), embedding_dim))
        # Going over glove embeddings, checking if the word is present in our vocab, 
        # if present find it's index and replace that index in the matrix with the embedding
        for line in f.readlines():
            values = line.split()    
            word = values[0]
            index = word2ind.get(word)
            if index:
                vector = np.array(values[1:], dtype='float32')
                embeddings[index] = vector
        return torch.from_numpy(embeddings).float()
    
class Vocabulary():
    
    def __init__(self, version=2):
        if version == 1:
            # use big vocab
            self.word2ind = pickle.load(open("word2inddict", "rb"))
            self.ind2word = pickle.load(open("ind2worddict", "rb"))
        else:
            # use small vocab
            self.word2ind = pickle.load(open("word2inddict2", "rb"))
            self.ind2word = pickle.load(open("ind2worddict2", "rb"))
        
    def __call__(self, word):
        if not word in self.word2ind:
            return self.word2ind['<unk>']
        return self.word2ind[word]
        
    def __len__(self):
        return len(self.ind2word.keys())
        
if __name__ == "__main__":
    # Run 'python build_vocab.py" to pickle the dictionaries so you can use Vocabulary 
    coco_train = COCO('./data/annotations/captions_train2014.json')
    with open('TrainIds.csv', 'r') as f:
        reader = csv.reader(f)
        train_ids = list(reader)

    train_ids = [int(i) for i in train_ids[0]]

    words = {}
    for img_id in train_ids:
        img_captions = coco_train.imgToAnns[img_id]
        captions = map(lambda x:x['caption'], img_captions)
#         [words.update(nltk.tokenize.word_tokenize(str(c).lower())) for c in captions]
        for c in captions:
            tokens = nltk.tokenize.word_tokenize(str(c).lower())
            for token in tokens:
                if token in words.keys():
                    words[token] = words[token] + 1
                else:
                    words[token] = 1
    ind = 3
    ind2word = {}
    for key, value in words.items():
        if value >= 5:
            ind2word[ind] = key
            ind += 1
#     ind2word = {i+3:key for i,(key, value) in enumerate(words.items()) if value >= 5}
#     ind2word = {i+3:v for i,v in enumerate(words)}
    ind2word[0] = '<start>'
    ind2word[1] = '<end>'
    ind2word[2] = '<unk>'
    
    word2ind = {v: k for k, v in ind2word.items()}
    print(len(ind2word))
    print(len(word2ind))

    pickle.dump(ind2word, open("ind2worddict2", "wb"))
    pickle.dump(word2ind, open("word2inddict2", "wb"))
            
            