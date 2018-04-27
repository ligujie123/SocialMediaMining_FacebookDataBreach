# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 18:16:31 2018

@author: pp
"""

import pandas as pd
from pymongo import MongoClient
import matplotlib.pyplot as plt
import numpy as np
import string
import nltk
import re
import time
from collections import defaultdict
from configparser import ConfigParser
from gensim import corpora, models, similarities
from nltk.tokenize import RegexpTokenizer
from string import digits

twitter=[]
client = MongoClient('localhost', 27017)
db = client.Facebook
for doc in db.tweets.distinct('text'):
    twitter.append(doc)

#remove url
twitter = [re.sub(r'https?:\/\/.*\/\w*','', doc) for doc in twitter ]

#remove tickers
twitter = [re.sub(r"(?:\@|http?\://)\S+", "", doc) for doc in twitter ]
#remove punctuation
twitter = [re.sub(r'[' + string.punctuation + ']+', ' ', doc) for doc in twitter]