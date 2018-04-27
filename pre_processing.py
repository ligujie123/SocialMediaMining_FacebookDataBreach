#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import re
from nltk.tokenize import word_tokenize, TweetTokenizer
from nltk.corpus import stopwords
from pymongo import MongoClient
import pandas as pd
from collections import Counter
from wordcloud import WordCloud
import pylab as pl

twitter=[]
client = MongoClient('localhost', 27017)
db = client.Facebook
for doc in db.tweets.distinct('text'):
    twitter.append(doc)

from nltk.corpus import stopwords
stop_words_english = stopwords.words('english')

def clean_tweet(tweet):
    # Remove tickers
    tweet = re.sub(r'\$\w*','', tweet)
    # Remove URLs
    tweet = re.sub(r'https?:\/\/.*\/\w*','', tweet)
    #Remove RT
    tweet = re.sub(r'RT','', tweet)
    # Remove puncutation
    tweet = re.sub(r'[' + string.punctuation + ']+', ' ', tweet)
    # Tokenize text
    tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)
    tokens = tokenizer.tokenize(tweet)
    # Remove stopwords and single characters
    tokens = [i.lower() for i in tokens if not i.lower() in stop_words_english and len(i) > 1]

    return tokens

cleaned_tweets = []
for tweet in twitter:
    cleaned_tweets.append(clean_tweet(tweet))

df = pd.DataFrame({'cleaned_text':cleaned_tweets})
words = [x for l in df.cleaned_text.values for x in l]
len(words)
c = Counter(words)

#
top = 20
most_common = c.most_common()[1:]
x, y = zip(*most_common[:top])

fig, ax = pl.subplots(1, figsize=(16,9))
ax.barh(range(len(x)), y)
ax.invert_yaxis()
ax.set_yticks(np.arange(len(x)) + 0.4)
ax.set_yticklabels(x, fontsize=12);
#
wc = WordCloud(relative_scaling=.5, width=800, height=300, background_color='white',
               max_words=1000).generate_from_frequencies(dict(c.most_common()[2:]))
fig = pl.figure(figsize=(16,8), frameon=False)
ax = pl.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
ax.imshow(wc, aspect='normal')
fig.savefig('text_wordcloud.png', dpi=300)

#sentiment
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer

twitter=[]
time=[]
client = MongoClient('localhost', 27017)
db = client.Facebook
for doc in db.tweets.find({},{"text":1,"created":1}):
    twitter.append(doc['text'])
    time.append(doc['created'])

def clean_tweet0(tweet):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())


cleaned_tweets = []
for tweet in twitter:
    cleaned_tweets.append(clean_tweet0(tweet))
df = pd.DataFrame({'text':cleaned_tweets, 'time':time})

def get_tweet_sentiment(tweet):
    analysis = TextBlob(tweet)
        # set sentiment
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity == 0:
        return 'neutral'
    else:
        return 'negative'

sentiment=[]
for i in range(len(df['text'])):
    sentiment.append(get_tweet_sentiment(df['text'][i]))

pos=0
neg=0
neu=0
for i in range(len(sentiment)):
    if sentiment[i]=="positive":
        pos = pos+1
    elif sentiment[i]=="negative":
        neg = neg+1
    else:
        neu = neu+1
df.insert(2, 'sentiment',sentiment)
df.to_csv('sentiment.csv')


#read the processed data (in Excel)

shr = pd.read_csv('sentiment_timeseries_byhour.csv', index_col='time')
shr.index = pd.to_datetime(shr.index)
shr.plot(grid=True, title="Sentiment Percentage by Hour", xticks=shr.index)

