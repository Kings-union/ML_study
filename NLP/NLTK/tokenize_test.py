import urllib2
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords

request = urllib2.Request('http://php.net/')
response = urllib2.urlopen(request)
html = response.read()
soup = BeautifulSoup(html, 'html5lib')
text = soup.get_text(strip=True)
tokens = text.split()
print (tokens)

clean_tokens = list()
sr = stopwords.words('english')
for token in tokens:
    if not token in sr:
        clean_tokens.append(token)

freq = nltk.FreqDist(clean_tokens)
# for key, val in freq.items():
#     print (str(key) + ':' + str(val))

freq.plot(20, cumulative=False)
