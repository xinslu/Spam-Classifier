import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from math import log, sqrt
import pandas as pd
import numpy as np
from wordcloud import WordCloud


def process_message(message, lower_case=True, stem=True, stop_words=True, gram=2):
    if lower_case:
        message = message.lower()
    words = word_tokenize(message)
    words = [w for w in words if len(w) > 2]
    if gram > 1:
        w = []
        for i in range(len(words) - gram + 1):
            w += [' '.join(words[i:i + gram])]
        return w
    if stop_words:
        sw=stopwords.words('english')
        words=[word for word in words if word not in sw]
    if stem:
        words = [PorterStemmer().stem(word) for word in words]
    return words


def word_graph(totalMails):
    trainIndex, testIndex = list(), list()
    for i in range(1, totalMails):
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

mails = pd.read_csv('.\input\spam.csv', encoding="latin-1")
mails.rename({'v1': 'label', 'v2': 'messages'}, axis=1, inplace=True)
mails.drop(mails.filter(regex="Unname"), axis=1, inplace=True)
totalMails = mails['messages'].shape[0]
print(mails['messages'].head().apply(process_message))
