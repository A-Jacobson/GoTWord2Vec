from gensim.models import word2vec
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import logging
import gensim
import os

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

text = open("data/corpus.txt").read().lower()

# remove newline characters
text = re.sub('\n+', "", text)

# Word2Vec accepts sequences of sentence so we will use nltk's punkt tokenizer to split our corpus into sentences
sentences = sent_tokenize(text)
sentences = [word_tokenize(s) for s in sentences]
print sentences[0]


model = gensim.models.Word2Vec(sentences, min_count=3, size=100, workers=4)
model.save(os.path.join("models", "GoT"))
