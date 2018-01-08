import re, collections

def words(text): return re.findall('[a-z]+', text.lower())

def train(feature):
    model = collections.defaultdict(lambda: 1)
    for f in feature:
        model[f] += 1
    return model

NWORDS = train(words(open('big.txt').read()))

alphabet = 'abcdefghijklmnopqrstuvwxyz'

def editsl(word):
    n = len(word)
    # print word, n
    return set([word[0:i]+word[i+1:] for i in range(n)] +
               [word[0:i]+word[i+1]+word[i]+word[i+2:] for i in range(n-1)] + # transposition
               [word[0:i]+c+word[i+1:] for i in range(n) for c in alphabet] + # alteration
               [word[0:i]+c+word[i:] for i in range(n+1) for c in alphabet])  # insertion


def known_edits2(word):
    return set(e2 for e1 in editsl(word) for e2 in editsl(e1) if e2 in NWORDS)

def known(words): return set(w for w in words if w in NWORDS)

def correct(word):
    candidates = known([word]) or known(editsl(word)) or known_edits2(word) or [word]
    return max(candidates, key=lambda w: NWORDS[w])

print correct('thi')