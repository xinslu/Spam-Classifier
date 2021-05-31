from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])


def split_data(mails, totalMails):
    trainIndex, testIndex = list(), list()
    for i in range(1, totalMails):
        if np.random.uniform(0, 1) < 0.75:
            trainIndex += [i]
        else:
            testIndex += [i]
    trainData = mails.iloc[trainIndex]
    testData = mails.iloc[testIndex]
    return trainData, testData


def word_graph(mails, val):
    spam_words = ' '.join(list(mails[mails['class'] == val]['messages']))
    spam_wc = WordCloud(width=512, height=512).generate(spam_words)
    plt.figure(figsize=(10, 8), facecolor='k')
    plt.imshow(spam_wc)
    plt.tight_layout(pad=0)
    plt.show()


stemmer = PorterStemmer()
mails = pd.read_csv('.\input\spam.csv', encoding="latin-1")
mails.rename({'v1': 'label', 'v2': 'messages'}, axis=1, inplace=True)
mails.drop(mails.filter(regex="Unname"), axis=1, inplace=True)
totalMails = mails['messages'].shape[0]
mails['class'] = mails['label'].map({'ham': 0, 'spam': 1})
mails.drop(['label'], axis=1, inplace=True)
[trainData, testData] = split_data(mails, totalMails)
# word_graph(mails, 1) #displays the word graph for spam
# word_graph(mails,0) #displays the word graph for ham
model = Pipeline([('vect', StemmedCountVectorizer(min_df=3, stop_words='english', ngram_range=(1, 2), tokenizer=word_tokenize, strip_accents='unicode')),
                  ('tfidf', TfidfTransformer()),
                  ('clf', LogisticRegression())])
model.fit(trainData['messages'], trainData['class'])
predicted = model.predict(testData['messages'])
print('Accuracy:', accuracy_score(testData['class'], predicted) * 100, "%")
# c1 = ['Congratulations ur awarded free $500'] prints 1
# content = pd.DataFrame(c1, columns=['Test Data'])
# print(model.predict(content['Test Data']))
