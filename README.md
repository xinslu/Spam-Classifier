# Spam Classifier
Build a spam classifier with uses the Naive-Bayesian TF-IDF (Term Frequency-Inverse Document Frequency) model.

# TF-IDF
TF-IDF stands for Term Frequency-Inverse Document Frequency. TF-IDF is a statistical measure that evaluates how relevant a word is to a document in a collection of documents. This is done by multiplying two metrics: how many times a word appears in a document, and the inverse document frequency of the word across a set of documents. I used a Tfidftransformer to transform the matrix to a tfidfcount matrix.

#Porter Stemmer
The Porter stemming algorithm is a process for removing the commoner morphological and inflexional endings from words in English. Its main use is as part of a term normalisation process that is usually done when setting up Information Retrieval systems. You can learn more here: https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html

#Tokenize
Given a character sequence and a defined document unit, tokenization is the task of chopping it up into pieces, called tokens , perhaps at the same time throwing away certain characters, such as punctuation. Learn more here: https://nlp.stanford.edu/IR-book/html/htmledition/tokenization-1.html

#N Grams
An N-gram means a sequence of N words. I used N Grams to split up the text and make it easier for the TF-IDF to categorize and evaluate. 

#Stop Words
Sometimes, some extremely common words which would appear to be of little value in helping select documents matching a user need are excluded from the vocabulary entirely. These words are called stop words .I used stop words to increase the effiency of the algorithm and make it easy to classify text by removing the unnecessary part of the text. Learn more here: https://nlp.stanford.edu/IR-book/html/htmledition/dropping-common-terms-stop-words-1.html

