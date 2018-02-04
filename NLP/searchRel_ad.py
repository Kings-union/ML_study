import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from nltk.stem.snowball import SnowballStemmer

df_train = pd.read_csv('train.csv',encoding="ISO-8859-1")
df_test = pd.read_csv('test.csv',encoding="ISO-8859-1")
df_desc = pd.read_csv('product_descriptions.csv',encoding="ISO-8859-1")

df_all = pd.concat((df_train,df_test),axis=0,ignore_index=True)
df_all = pd.merge(df_all,df_desc,how='left',on='product_uid')
# text pre-processing
stemmer = SnowballStemmer('english')

def str_stemmer(s):
    return " ".join([stemmer.stem(word) for word in s.lower().split()])

def str_common_word(str1,str2):
    return sum(int(str2.find(word)>=0) for word in str1.split())

df_all['search_term'] = df_all['search_term'].map(lambda x:str_stemmer(x))
df_all['product_title'] = df_all['product_title'].map(lambda x:str_stemmer(x))
df_all['product_description'] = df_all['product_description'].map(lambda x:str_stemmer(x))

# Advanced version

# pip install python-Levenshtein
import Levenshtein

df_all['dist_in_title'] = df_all.apply(lambda x:Levenshtein.ratio(x['search_term'],x['product_title']),axis=1)
df_all['dist_in_desc'] = df_all.apply(lambda x:Levenshtein.ratio(x['search_term'],x['product_description']),axis=1)

# TF-IDF
df_all['all_texts'] = df_all['product_title'] + ' . ' + df_all['product_description'] + ' . '
print(df_all['all_texts'][:5])

from gensim.utils import tokenize
from gensim.corpora.dictionary import Dictionary
dictionary = Dictionary(list(tokenize(x,errors='ignore')) for x in df_all['all_texts'].values)
print(dictionary)

class MyCorpus(object):
    def __iter__(self):
        for x in df_all['all_texts'].values:
            yield dictionary.doc2bow(list(tokenize(x,errors='ignore')))

corpus = MyCorpus()

from gensim.models.tfidfmodel import TfidfModel
tfidf = TfidfModel(corpus)
# print(tfidf[dictionary.doc2bow(list(tokenize('hello world, good morning', errors='ignore')))])

# calculate the similiarity for the sentence
from gensim.similarities import MatrixSimilarity

# package each sentence
def to_tfidf(text):
    res = tfidf[dictionary.doc2bow(list(tokenize(text,errors='ignore')))]
    return res

def cos_sim(text1,text2):
    tfidf1 = to_tfidf(text1)
    tfidf2 = to_tfidf(text2)
    index = MatrixSimilarity([tfidf1],num_features=len(dictionary))
    sim = index[tfidf2]
    return  float(sim(0))

# text1 = 'hello world'
# text2 = 'hello from the other side'
# print(cos_sim(text1,text2))

df_all['tfidf_cos_sim_in_title'] = df_all.apply(lambda x:cos_sim(x['search_term'],x['product_title']),axis=1)
df_all['tfidf_cos_sim_in_desc'] = df_all.apply(lambda x:cos_sim(x['search_term'],x['product_description']),axis=1)

import nltk
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
# tokenizer.tokenize(df_all['all_texts'].values[0])

# split the text to sentence and flatten
sentences = [tokenizer.tokenize(x) for x in df_all['all_texts'].values]
sentences = [y for x in sentences for y in x]

# split the sentence to words
from nltk.tokenize import word_tokenize
w2v_corpus = [word_tokenize(x) for x in sentences]

# word2vec
from gensim.models.word2vec import Word2Vec
model = Word2Vec(w2v_corpus,size=128,window=5,min_count=5,workers=4)
vocab = model.wv.vocab

def get_vector(text):
    res = np.zeros([128])
    count = 0
    for word in word_tokenize(text):
        if word in vocab:
            res += model[word]
            count += 1
    return res/count

# calculate the similarity for each text
from scipy import spatial

def w2v_cos_sim(text1,text2):
    try:
        w2v1 = get_vector(text1)
        w2v2 = get_vector(text2)
        sim = 1 - spatial.distance.cosine(w2v1,w2v2)
        return float(sim)
    # in case we will get the 0 vector
    except:
        return float(0)

df_all['w2v_cos_sim_in_title'] = df_all.apply(lambda x: w2v_cos_sim(x['search_term'], x['product_title']), axis=1)
df_all['w2v_cos_sim_in_desc'] = df_all.apply(lambda x: w2v_cos_sim(x['search_term'], x['product_description']), axis=1)
# delete the uplearning attribute
df_all = df_all.drop(['search_term','product_title','product_description','all_texts'],axis=1)

# rebuild the train/test set
df_train = df_all.loc[df_train.index]
df_test = df_all.loc[df_test.index]
test_ids = df_test['id']

# split the train result
Y_train = df_train['relevance'].value
X_train = df_train.drop(['id','relevance'],axis=1).values
X_test = df_test.drop(['id','relevance'],axis=1).values

# build the model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

params = [1,3,5,6,7,8,9,10]
test_scores = []
for param in params:
    clf = RandomForestRegressor(n_estimators=30, max_depth=param)
    test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=5, scoring='neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))

# draw the pic for the CV error:
import matplotlib.pyplot as plt
plt.plot(params,test_scores)
plt.title("Param vs CV Errors")
plt.show()

# upload the result:
rf = RandomForestRegressor(n_estimators=30,max_depth=6)
rf.fit(X_train,Y_train)

Y_pred = rf.predict(X_test)
pd.DataFrame({"id": test_ids, "relevance": Y_pred}).to_csv('submission.csv',index=False)