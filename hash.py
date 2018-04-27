# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 09:44:26 2018

@author: pp
"""

import pandas as pd
from pymongo import MongoClient
import matplotlib.pyplot as plt
import numpy as np

hashtag=[]
client = MongoClient('localhost', 27017)
db = client.Facebook
for doc in db.tweets.find({},{"hashtags":1}):
    hashtag.append(doc['hashtags'])

length=[]
for i in range(len(hashtag)):
    i = len(hashtag[i])
    length.append(i)

result=[1]*len(hashtag)
for i in range(len(hashtag)):
    result[i] = []

for i in range(len(hashtag)):
    for j in range(length[i]):
        result[i].append(hashtag[i][j]['text'].lower())

hashtags = pd.DataFrame({'hashtags': result, 'number': length})

plt.hist(hashtags['number'],bins = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
plt.xlabel('# of hashtag')
plt.ylabel('Frequency')
plt.title('Histogram of hashtag')
plt.xlim(0,15)
plt.grid(True)
plt.show()

#
hashtag_1 = []
for i in range(len(hashtag)):
    for j in range(length[i]):
        hashtag_1.append(hashtag[i][j]['text'].lower())
hashtag_1 = pd.DataFrame({'hashtag_1': hashtag_1})
grouped = hashtag_1.groupby(['hashtag_1']).size()
grouped = grouped.order(ascending=False)
top30 = grouped[0:30]
number = []
for i in range(30):
    number.append(top30[i])
top30 = pd.DataFrame({'hashtag':top30.index, 'number':number})
x=np.arange(30)
plt.figure(figsize=(13,4))
plt.bar(x, top30['number'].values,alpha=0.7)
plt.xticks(x, top30['hashtag'].values, rotation=65, size=9)
plt.title('Top30 Hashtags')
plt.grid(True)
plt.show()

####
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

hashtag=[]
client = MongoClient('localhost', 27017)
db = client.Facebook
for doc in db.tweets.find({},{"hashtags":1}):
    hashtag.append(doc['hashtags'])

length=[]
for i in range(len(hashtag)):
    i = len(hashtag[i])
    length.append(i)

result=[1]*len(hashtag)
for i in range(len(hashtag)):
    result[i] = []

for i in range(len(hashtag)):
    for j in range(length[i]):
        result[i].append(hashtag[i][j]['text'].lower())

tran=[]
for i in range(len(result)):
    if len(result[i]) > 1:
        tran.append(result[i])

te = TransactionEncoder()
te_ary = te.fit(tran).transform(tran)
df = pd.DataFrame(te_ary, columns=te.columns_)
frequent_itemsets = apriori(df, min_support = 0.012, use_colnames=True)

from mlxtend.frequent_patterns import association_rules
finalresult = association_rules(frequent_itemsets, metric = "lift", min_threshold=0.6)
finalresult.to_csv("associationrule.csv")

#
cooc=[]
for i in range(len(tran)):
    if len(tran[i]) == 2:
        cooc.append(tran[i])
    else:
        for j in range(len(tran[i])-1):
            k = j+1
            while k<=len(tran[i])-1:
                cooc.append([tran[i][j],tran[i][k]])
                k = k+1
cooc = pd.DataFrame(cooc)
cooc.to_csv("cooc.csv", encoding='utf-8')

#
pl = pd.read_csv('pl.csv', header=None, sep=',', names=['degree', 'prob'])
plt.scatter(pl['degree'], pl['prob'], marker='o', alpha=0.8)
plt.title('Hashtag Co-occurrence Network Degree Distribution (log-log)')
plt.show()