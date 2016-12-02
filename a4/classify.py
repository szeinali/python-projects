from collections import Counter
import matplotlib.pyplot as plot
import networkx as nx
import sys
import time
from TwitterAPI import TwitterAPI
import json
from collections import defaultdict
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen


def load_file():
    with open('result2') as data_file:    
        data = json.load(data_file)


url = urlopen('http://www2.compute.dtu.dk/~faan/data/AFINN.zip')
zipfile = ZipFile(BytesIO(url.read()))
afinn_file = zipfile.open('AFINN/AFINN-111.txt')

afinn = dict()

for line in afinn_file:
    parts = line.strip().split()
    if len(parts) == 2:
        afinn[parts[0].decode("utf-8")] = int(parts[1])


def afinn_sentiment2(terms, afinn, verbose=False):
    pos = 0
    neg = 0
    for t in terms:
        if t in afinn:
            if verbose:
                print('\t%s=%d' % (t, afinn[t]))
            if afinn[t] > 0:
                pos += afinn[t]
            else:
                neg += -1 * afinn[t]
    return pos, neg


tweet_pos=0
tweet_neg=0
for i in range(len(tweet)):
    terms=tweet[i]['text'].split()
    pos,neg=afinn_sentiment2(terms, afinn, verbose=False)
    if pos>neg:
        tweet_pos=tweet_pos+1
    else:
        tweet_neg=tweet_neg+1
total=tweet_pos+tweet_neg

print("People in Support of Narendra Modi:"+str((tweet_pos*100)/total)+"%")
print("People Against of Narendra Modi:"+str((tweet_neg*100)/total)+"%")



def main():
    load_file()
    afinn_sentiment2(terms, afinn, verbose=False)

if __name__ == '__main__':
    main()
