import pickle
import nltk
from pycocotools.coco import COCO
from collections import defaultdict
import csv

class Vocabulary():
    
    def __init__(self):
        self.word2ind = pickle.load(open("word2inddict", "rb"))
        self.ind2word = pickle.load(open("ind2worddict", "rb"))
        
    def __call__(self, word):
        try:
            return self.word2ind[word]
        except:
            return 99999
        
    def __len__(self):
        return len(self.ind2word.keys())
        
if __name__ == "__main__":
    # Run 'python build_vocab.py" to pickle the dictionaries so you can use Vocabulary 
    coco_train = COCO('./data/annotations/captions_train2014.json')
    with open('TrainImageIds.csv', 'r') as f:
        reader = csv.reader(f)
        train_ids = list(reader)

    train_ids = [int(i) for i in train_ids[0]]

    words = set()
    for img_id in train_ids:
        img_captions = coco_train.imgToAnns[img_id]
        captions = map(lambda x:x['caption'], img_captions)
        [words.update(nltk.tokenize.word_tokenize(str(c).lower())) for c in captions]
        
#     words.update(['<start>', '<end>'])
    
    ind2word = {i+2:v for i,v in enumerate(words)}
    ind2word[0] = '<start>'
    ind2word[1] = '<end>'
    ind2word[99999] = '<unk>'
    
    word2ind = {v: k for k, v in ind2word.items()}
    
    pickle.dump(ind2word, open("ind2worddict", "wb"))
    pickle.dump(word2ind, open("word2inddict", "wb"))
            
            