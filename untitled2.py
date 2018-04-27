# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 22:18:14 2018

@author: pp
"""
import pandas as pd
from pymongo import MongoClient

hashtags = []
client = MongoClient('localhost', 27017)
db = client.Facebook
for doc in db.tweets.find({},{"hashtags":1}):
    hashtags.append(doc['hashtags'])