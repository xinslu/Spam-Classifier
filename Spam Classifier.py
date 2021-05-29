from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from math import log, sqrt
import pandas as pd
import numpy as np
from wordcloud import WordCloud

mails = pd.read_csv('.\input\spam.csv', encoding="latin-1")
mails.rename({'v1': 'label', 'v2': 'messages'}, axis=1, inplace=True)
mails.drop(mails.filter(regex="Unname"), axis=1, inplace=True)
totalMails = mails['messages'].shape[0]
trainIndex, testIndex = list(), list()
for i in range(1, mails.shape[0]):
    if np.random.uniform(0, 1) < 0.75:
        trainIndex += [i]
    else:
        testIndex += [i]
trainData = mails.loc[trainIndex]
testData = mails.loc[testIndex]
spam_words = ' '.join(list(mails[mails['label'] == 'spam']['messages']))
spam_wc = WordCloud(width=512, height=512).generate(spam_words)
plt.figure(figsize=(10, 8), facecolor='k')
plt.imshow(spam_wc)
plt.tight_layout(pad=0)
plt.show()
