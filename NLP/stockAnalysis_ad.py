'''
In this case, we will use work2vec for the nlp analysis
'''

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from datetime import date

data = pd.read_csv('Combined_News_DJIA.csv')

train = data[data['Date'] < '2015-01-01']
test = data[data['Date'] > '2014-12-31']

X_train = train[train.columns[2:]]
# corpus is for the list resource
corpus = X_train.values.flatten().astype(str)
X_train = X_train.values.astype(str)
X_train = np.array([' '.join(x) for x in X_train])
X_test = test[test.columns[2:]]
X_test = X_test.values.astype(str)
X_test = np.array([' '.join(x) for x in X_test])
Y_train = train['Label'].values
Y_test = test['Label'].values

# split each work for the sentence
from nltk.tokenize import word_tokenize
corpus = [word_tokenize(x) for x in corpus]
X_train = [word_tokenize(x) for x in X_train]
X_test = [word_tokenize(x) for x in X_test]

# pre-process the work
# remove stopword, number and formate the word

from nltk.corpus import stopwords
stop = stopwords.words('english')

import re
def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))

def isSymbol(inputString):
    return bool(re.match(r'[^\w]',inputString))

# lemma(format the word)
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

def check(word):
    if word in stop:
        return False
    elif hasNumbers(word) or isSymbol(word):
        return False
    else:
        return True

def preprocessing(sens):
    res = []
    for word in sens:
        if check(word):
            word = word.lower().replace("b'",'').replace('b"','').replace('"','').replace("'",'')
            res.append(wordnet_lemmatizer.lemmatize(word))
    return res

corpus = [preprocessing(x) for x in corpus]
X_train = [preprocessing(x) for x in X_train]
X_test = [preprocessing(x) for x in X_test]

from gensim.models.word2vec import Word2Vec

# we use 128 vector for the word represent
model = Word2Vec(corpus,size=128,window=5,min_count=5,workers=4)

vocab = model.wv.vocab

def get_vector(word_list):
    res = np.zeros([128])
    count = 0
    for word in word_list:
        if word in vocab:
            res += model[word]
            count += 1
    return res/count

wordlist_train = X_train
wordlist_test = X_test

X_train = [get_vector(x) for x in X_train]
X_test = [get_vector(x) for x in X_test]

# build the ML model

from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score

params = [0.1,0.5,1,3,5,7,10,12,16,20,25,30,35,40]
test_scores = []
for param in params:
    clf = SVR(gamma=param)
    test_score = cross_val_score(clf,X_train,Y_train,cv=3,scoring='roc_auc')
    test_scores.append(np.mean(test_score))

import matplotlib.pyplot as plt
plt.plot(params,test_scores)
plt.title("Param vs CV AUC Score")
plt.show()

