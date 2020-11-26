# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import re
import string
import spacy
import pandas as pd
import numpy as np
import warnings
import textacy.ke
import matplotlib.pyplot as plt
# %matplotlib inline
warnings.filterwarnings('ignore')

from tabulate import tabulate
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec, KeyedVectors
from spellchecker import SpellChecker
from spacy import displacy
from spacy.matcher import Matcher


# -

# Cleaning up the corpus
def cleanup(text):    
    text = text.lower() # lowers the corpus
    text = re.sub("n't", '', str(text)) # removes not with apostrophe
    text = re.sub("'\w+|’\w+", '', str(text)) # removes contraction
    text = re.sub('-(?<!\d)', ' ', str(text)) # removing hyphens from numbers
    custom_punctuation = punctuation.replace(".", "") # create custom punctuation list
    text = text.translate(str.maketrans('', '', custom_punctuation)) # removes all punctuation except period
    text = re.sub("\.{2,}", '', str(text)) # removes "..."
    text = ' '.join([token for token in text.split()]) # removes trailing whitespaces
    text = word_tokenize(text) # tokenize words
    stopwords_extra = ['na', 'u', 'id', 'gon', 'pas', '10184285', '179180', 'as', 'oh', 'wan', 'av', 'p', 'ta', '10000', '6000']
    text = [word for word in text if not word in stopwords_extra] # remove custome stopwords
    text = ' '.join(text) # join the words back together  
    return text


# loading the corpus data
df = pd.read_csv('rleaves.csv', encoding='utf-8')

# apply preprocessing function
corpus = pd.DataFrame(df['raw'].apply(cleanup))

# create a new series separated by rows by sentences
sentokenized = corpus['raw'].str.split('.').apply(pd.Series, 1).stack()
sentokenized.index = sentokenized.index.droplevel(-1)
sentokenized.name = 'raw'
sentokenized = pd.DataFrame(sentokenized)

# selecting only the rows with time value
sentokenized['num'] = pd.np.where(sentokenized.raw.str.contains('\d+'), True, False)
sentokenized['time'] = pd.np.where(sentokenized.raw.str.contains('\s*days* |\s*months* |\s*weeks* |\s*ye?a?rs* '), True, False)
rleaves = sentokenized.loc[(sentokenized['num'] == True) & (sentokenized['time'] == True)].reset_index().rename(columns={'index': "post"})
rleaves['totalwords'] = rleaves['raw'].str.split().str.len()
rleaves['day'] = pd.np.where(rleaves.raw.str.contains('\d+\s*day[s\s]|\s*day\s*\d+'), True, False)
rleaves['week'] = pd.np.where(rleaves.raw.str.contains('\d+\s*week[s\s]|\s*week\s*\d+'), True, False)
rleaves['month'] = pd.np.where(rleaves.raw.str.contains('\d+\s*month[s\s]|\s*month\s*\d+'), True, False)
rleaves['year'] = pd.np.where(rleaves.raw.str.contains('\d+\s*ye?a?r[s\s]*'), True, False)

no_dates = rleaves.loc[(rleaves['day'] == False) & (rleaves['week'] == False) & (rleaves['month'] == False) & (rleaves['year'] == False)].reset_index(drop=True)

nlp = spacy.load('en_core_web_sm')
doc = nlp('we now get over 145000 unique users a month here at rleaves so even a small participation can help a whole lot of people')

for ent in doc.ents:
    print(ent.text, ent.label_)

# +
import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")
text = 'you also realise you are about £100 better off every single week which is the equivalent of getting a 5 grand pay rise at work tell me that isn a good incentive eh here the secret the difficult part of quitting is getting yourself to a place mentally where you are 100 ready to leave weed behind and move on with your life'
doc = nlp(text)
displacy.render(doc, style='ent')

# + jupyter={"outputs_hidden": true}
str(rleaves.raw.tolist())

# +
import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")

text = str(rleaves.raw.tolist())
doc = nlp(text)

matcher = Matcher(nlp.vocab)
pattern = [{'LIKE_NUM': True}, {'LOWER': {'REGEX' : 'day'}}]
matcher.add('TIME', None, pattern)

matches = matcher(doc)
print([doc[start:end] for match_id, start, end in matches])
#=> [5am, 6am, 9pm, 6am, 6am, 9pm]
# -

def validate(text):
    nlp = spacy.load('en_core_web_sm')
    docs = list(nlp.pipe(text))
    entities = [doc.ents for doc in docs]
    options = {"ents": ['DATE']}
    displacy.render(docs, style="ent", options=options)


rleaves.raw

validate(rleaves.raw)

no_dates.raw.values.tolist()

# load english language model
nlp = spacy.load('en_core_web_sm')
matcher = Matcher(nlp.vocab)


def find_days(text):
    pattern = [{'DEP': 'pobj'},
               {'POS': 'NOUN'}]
    #pattern = [{'DEP': 'ccomp'}]
    matcher.add("test", None, pattern)
    matches = matcher(text)
    print('Total matches found:', len(matches))
    for match_id, start, end in matches:
        span = doc[start:end]
        print(span.text)    


find_days(doc)

rleaves.raw[541]

doc = nlp(' '.join(rleaves.raw))

# !jupytext --to py BS-FeatureEngineering.ipynb


