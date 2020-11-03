import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import re
import string     
from pprint import pprint
from bs4 import BeautifulSoup

import nltk
from nltk.tokenize import word_tokenize, WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords

import spacy

#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')

stop_words = stopwords.words('english')
sp = spacy.load('en_core_web_sm')
lemmatizer = WordNetLemmatizer()


def make_dataset(datasets_pd_array):
    # union all datasets
    df_all = pd.concat(datasets_pd_array, ignore_index=True)
    df_all['storypoint'] = df_all['storypoint'].astype(str)
    # valid label names
    label_names = ['1', '2', '3', '5', '8', '13', '20', '40', '100']
    
    # new dataframe with valid datapoints
    df_all = df_all.loc[df_all['storypoint'].isin(label_names)]
    
    # seperate data from labels
    X = df_all.drop(["issuekey", "storypoint"], axis=1)
    y = df_all["storypoint"].values
    
    # replace missing values with empty strings
    X = X.fillna('')
    
    # concat title with description
    X["title_description"] = X["title"] + " " +  X["description"]
    
    # get numpy array
    text = X["title_description"].to_numpy()

    return text, y


def preprocess_txt(doc):
    #lowercase
    doc = doc.lower() 
    #remove "{html}" strings
    doc = re.sub('\{html\}', '', doc)
    #remove html tags
    doc = BeautifulSoup(doc, 'html.parser').get_text()
    #remove all paths/urls/--keys
    pattern = re.compile(r'[/\-+\\+]')
    doc_split = [token for token in WhitespaceTokenizer().tokenize(doc) if not pattern.findall(token)]
    doc = " ".join(doc_split)

    #tokenize and remove stop words and punctuation symbols and spaces using spaCy
    #use lemmas
    doc_spacy = sp(doc)
    doc_tokenized_spacy = [token.lemma_ for token in doc_spacy
        if not token.is_stop and not token.is_punct and not token.is_space]

    #preprocessing additionaly with nltk give much better results
    doc_nltk = " ".join(doc_tokenized_spacy)
    #tokenize and remove stop words and punctuation symbols using nltk 
    #remove numerics
    doc_tokenized_spacy_nltk = [token for token in nltk.word_tokenize(doc_nltk)
        if token.isalpha()]
    
    return doc_tokenized_spacy_nltk 


def create_corpus(text_arr):
    corpus = []

    for doc in text_arr:
        prep_doc = preprocess_txt(doc)                            
        corpus.append(prep_doc)
        
    return corpus  

def create_vocab(corpus):
    vocab = set(token for doc in corpus for token in doc)
    
    return vocab

def show_most_freq_n(corp, n):
    allwords = [token for doc in corp for token in doc]
    
    # word freqs    
    mostcommon_small = nltk.FreqDist(allwords).most_common(n)
    print(mostcommon_small)
    x, y = zip(*mostcommon_small)
    plt.figure(figsize=(50,30))
    plt.margins(0.02)
    plt.bar(x, y)
    plt.xlabel('Words', fontsize=50)
    plt.ylabel('Frequency of Words', fontsize=50)
    plt.yticks(fontsize=40)
    plt.xticks(rotation=60, fontsize=40)
    plt.title('Frequency of {} Most Common Words'.format(n), fontsize=60)
    plt.show()    


