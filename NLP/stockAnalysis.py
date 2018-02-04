from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from datetime import date

data = pd.read_csv('Combined_News_DJIA.csv')

# print the data.head() to check the data structure
#print(data.head())

data["combined_news"] = data.filter(regex=("Top.*")).apply(lambda x:''.join(str(x.values)),axis=1)

# print(data["Combined_news"])

train = data[data['Date'] < '2015-01-01']
test = data[data['Date'] > '2014-12-31']

feature_extraction = TfidfVectorizer()


X_train = train["combined_news"].str.lower().str.replace('"','').str.replace(",",'').str.split()
X_test = test["combined_news"].str.lower().str.replace('"','').str.replace(",",'').str.split()
# print(X_test[1611])

# remove stop word
from nltk.corpus import stopwords
stop = stopwords.words('english')
# remove number
import re
def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))

# lemma(format the word)
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

def check(word):
    if word in stop:
        return False
    elif hasNumbers(word):
        return False
    else:
        return True

X_train = X_train.apply(lambda x: [wordnet_lemmatizer.lemmatize(item) for item in x if check(item)])
X_test = X_test.apply(lambda x: [wordnet_lemmatizer.lemmatize(item) for item in x if check(item)])

X_train = X_train.apply(lambda x: ' '.join(x))
X_test = X_test.apply(lambda x:' '.join(x))


X_train = feature_extraction.fit_transform(train["combined_news"].values)
X_test = feature_extraction.transform(test["combined_news"].values)

Y_train = train["Label"].values
Y_test = test["Label"].values



# SVC means classifier of SVM, SVR means Regression of SVM, in this case, 0/1 problem need SVC
clf = SVC(probability=True,kernel='rbf')
clf.fit(X_train,Y_train)
predictions = clf.predict_proba(X_test)

print('ROC-AUC yields ' + str(roc_auc_score(Y_test,predictions[:,1])))
