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
    
    # lowercase the corpus 
    text = text.lower()
    # removing apostrophes
    text = re.sub("'s", '', str(text))
    # removing hyphens from numbers
    #text = re.sub('-(?<!\d)', ' ', str(text))
    # removing punctuation
    custom_punctuation = punctuation.replace(".", "")
    text = text.translate(str.maketrans('', '', custom_punctuation))
    text = re.sub("\.{2,}", '', str(text))
    # removing emoticons
    #text = re.sub('[^\w\s,]', ' ', str(text))
    # removing zero-width space characters
    text = re.sub('x200b', ' ', str(text))
    # removing trailing whitespaces
    text = ' '.join([token for token in text.split()])
    # word tokenization
    text = word_tokenize(text)
    # additional removal of unnecessary words
    stopwords_extra = ['im', 'ive', 'dont', 'didnt', 'doesnt', 'isnt', 
                       'couldnt', 'na', 'youre', 'cant', 'u', 'id', 'wasnt', 
                       'gon', 'pas', 'ill', 'youve', 'wont', 'havent', 
                       'wouldnt', '10184285', '179180', 'arent', 'youll', 'as', 
                       'oh', 'wan', 'av', 'p', 'ta', '10000', '6000']
    text = [word for word in text if not word in stopwords_extra]
    # join the words
    text = ' '.join(text)

    return text


# Cleaning up the corpus
def rulebased(text):
    text = sent_tokenize(text)
   
    return text


df = pd.read_csv('rleaves.csv', encoding='utf-8')

# apply preprocessing function
df = pd.DataFrame(df['raw'].apply(cleanup))

df = pd.DataFrame(df['raw'].apply(rulebased))

df['has_numbers'] = pd.np.where(df.raw.str.contains('\d+'), True, False)

df['has_time'] = pd.np.where(df.raw.str.contains('\s*days* |\s*months* |\s*weeks* |\s*ye?a?rs* '), True, False)

rleaves = df.loc[(df['has_time'] == True) & (df['has_numbers'] == True)].reset_index(drop=True)

# + jupyter={"source_hidden": true}
#nlp = spacy.load("en_core_web_sm")
#matcher = Matcher(nlp.vocab)
#
#doc = nlp(rleaves.raw[0])
#
#pattern = [{'DEP': 'acl'}, {'POS': 'NOUN'}]
#matcher.add("numeric modifier", None, pattern)
#matches = matcher(doc)
#for match_id, start, end in matches:
#    # Get the matched span
#    matched_span = doc[start:end]
#    print(matched_span.text)
#
#def set_sentiment(matcher, doc, i, matches):
#    doc.sentiment += 0.1
#
#pattern1 = [{"ORTH": "Google"}, {"ORTH": "I"}, {"ORTH": "/"}, {"ORTH": "O"}]
#pattern2 = [[{"ORTH": emoji, "OP": "+"}] for emoji in ["😀", "😂", "🤣", "😍"]]
#matcher.add("GoogleIO", None, pattern1)  # Match "Google I/O" or "Google i/o"
#matcher.add("HAPPY", set_sentiment, *pattern2)  # Match one or more happy emoji
#
#doc = nlp("A text about Google I/O 😀😀")
#matches = matcher(doc)
#
#for match_id, start, end in matches:
#    string_id = nlp.vocab.strings[match_id]
#    span = doc[start:end]
#    print(string_id, span.text)
#print("Sentiment", doc.sentiment)
# -

sentokenized = rleaves['raw'].str.split('.').apply(pd.Series, 1).stack()

sentokenized.index = sentokenized.index.droplevel(-1)

sentokenized.name = 'raw'

del rleaves['raw']

rleaves = rleaves.join(sentokenized)

rleaves.shape

rleaves['has_numbers1'] = pd.np.where(rleaves.raw.str.contains('\d+'), True, False)
rleaves['has_time1'] = pd.np.where(rleaves.raw.str.contains('\s*days* |\s*months* |\s*weeks* |\s*ye?a?rs* '), True, False)

new_rleaves = rleaves.loc[(rleaves['has_time1'] == True) & (rleaves['has_numbers1'] == True)].reset_index(drop=True).drop(columns=['has_time', 'has_numbers'])

new_rleaves

# !jupytext --to py BS-TextExtraction.ipynb
