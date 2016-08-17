# from gensim.models import word2vec
from nltk.tokenize import sent_tokenize
import re
import logging
import gensim

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

text = open("data/corpus.txt").read().lower()
text = re.sub('\n+', "", text)

sentences = sent_tokenize(text)

model = gensim.models.Word2Vec(sentences, min_count=1, size=100, workers=4)
