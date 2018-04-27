# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 15:20:30 2018

@author: pp
"""

from gensim.models import word2vec
from nltk.corpus import stopwords
stop_words_english = stopwords.words('english')
import gensim
import logging
import tempfile
import re
from nltk.tokenize import word_tokenize, TweetTokenizer
import os
import string
punc = string.punctuation
corpus=[]
client = MongoClient('localhost', 27017)
db = client.Facebook
for doc in db.tweets.distinct('text'):
    corpus.append(doc)

def clean_tweet(tweet):
    # Remove tickers
    tweet = re.sub(r'\$\w*','', tweet)
    # Remove URLs
    tweet = re.sub(r'https?:\/\/.*\/\w*','', tweet)
    #Remove RT
    tweet = re.sub(r'RT','', tweet)
    # Remove puncutation
    tweet = re.sub(r'[' + punc + ']+', ' ', tweet)
    # Tokenize text
    tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)
    tokens = tokenizer.tokenize(tweet)
    # Remove stopwords and single characters
    tokens = [i.lower() for i in tokens if not i.lower() in stop_words_english and len(i) > 1]
    return tokens

texts = []
for tweet in corpus:
    texts.append(clean_tweet(tweet))
import pandas as pd
text = pd.DataFrame(texts)
text.to_csv('tweets.txt', sep='\t', index=False, encoding='utf-8')

sentences = word2vec.Text8Corpus('tweets.txt')
model = word2vec.Word2Vec(sentences, size=50)

for i in model.most_similar('privacy'):
    print i[0], i[1]

for i in model.most_similar('data'):
    print i[0], i[1]

for i in model.most_similar('trump'):
    print i[0], i[1]

for i in model.most_similar('facebook'):
    print i[0], i[1]

for i in model.most_similar('mark'):
    print i[0], i[1]

for i in model.most_similar('zuckerberg'):
    print i[0], i[1]

for i in model.most_similar('markzuckerberg'):
    print i[0], i[1]

for i in model.most_similar('usa'):
    print i[0], i[1]
    
for i in model.most_similar('google'):
    print i[0], i[1]

for i in model.most_similar('twitter'):
    print i[0], i[1]

for i in model.most_similar('cambridgeanalytics'):
    print i[0], i[1]

for i in model.most_similar('congress'):
    print i[0], i[1]

model.save('tiwtterw2v.model')
model.save_word2vec_format('twitter.model.bin', binary=True)