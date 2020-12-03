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
# library needed for this project
import re
import string
import spacy
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from spacy import displacy
from spacy.matcher import Matcher
# -

# loading the corpus data
df = pd.read_csv('rleaves.csv', encoding='utf-8')


# +
# custom function to clean reddit posts
def cleanup(text):    
    text = text.lower() # lowers the corpus
    text = re.sub("'\w+|’\w+", '', str(text)) # removes contraction
    text = re.sub('-(?<!\d)', ' ', str(text)) # removing hyphens from numbers
    custom_punctuation = punctuation.replace(".", "") # create custom punctuation list
    text = text.translate(str.maketrans('', '', custom_punctuation)) # removes all punctuation except period
    text = re.sub("\.{2,}", '', str(text)) # removes "..."
    text = ' '.join([token for token in text.split()]) # removes trailing whitespaces
    text = word_tokenize(text) # tokenize words
    stopwords_extra = ['im', 'ive', 'didnt', 'na', 'u', 'id', 'gon', 'pas', 'ill', 'wont', 'arent', 'as', 'oh', 'wan', 'av', 'p', 'ta', '10000', '6000']
    text = [word for word in text if not word in stopwords_extra] # remove custome stopwords
    text = ' '.join(text) # join the words back together  
    return text

# apply the function
df_clean = df['raw'].apply(cleanup)
# -

# create a new series separated by rows by sentences
sentokenized = df_clean.str.split('.').apply(pd.Series, 1).stack() # flatten the data 
sentokenized.index = sentokenized.index.droplevel(-1) # remove old index
sentokenized.name = 'raw' # name the series
sentokenized = pd.DataFrame(sentokenized).reset_index(drop=True)

# +
# lemmatizing and extra clean up using spacy
texts = sentokenized.raw.tolist()
clean = []
nlp = spacy.load('en_core_web_sm')

for doc in nlp.pipe(texts, batch_size=50, n_process=3):
    clean.append([t.lemma_ for t in doc if not t.is_stop and not t.is_punct and not t.like_url if t.is_ascii])

cleaner = pd.Series([' '.join(map(str, l)) for l in clean], name='raw')
# -

# Create new columns indicating period of time in sentences
rleaves = pd.DataFrame(cleaner)
rleaves['num'] = pd.np.where(rleaves.raw.str.contains('\d+'), True, False) # looks for all numbers
rleaves['time'] = pd.np.where(rleaves.raw.str.contains('\s*days* |\s*months* |\s*weeks* |\s*ye?a?rs* '), True, False) # looks for time
rleaves = rleaves.loc[(rleaves['num'] == True) & (rleaves['time'] == True)].reset_index().rename(columns={'index': "post"}) # filter only those rows
rleaves['totalwords'] = rleaves['raw'].str.split().str.len() # counts number of words per sentences
rleaves['day'] = pd.np.where(rleaves.raw.str.contains('\d+\s*day[s\s]|\s*day\s*\d+'), True, False) # indicates weather number is day or not
rleaves['week'] = pd.np.where(rleaves.raw.str.contains('\d+\s*week[s\s]|\s*week\s*\d+'), True, False) # indicates weather number is week or not
rleaves['month'] = pd.np.where(rleaves.raw.str.contains('\d+\s*month[s\s]|\s*month\s*\d+'), True, False) # indicates weather number is month or not
rleaves['year'] = pd.np.where(rleaves.raw.str.contains('\d+\s*ye?a?r[s\s]*'), True, False) # indicates weather number is year or not
rleaves.drop(columns=['num', 'time'], inplace=True)

# +
# create a column with all the adjectives
ADJ = []
VERB = []

for doc in nlp.pipe(rleaves.raw.tolist(), batch_size=50, n_process=3):
    ADJ.append([t.text for t in doc if t.pos_ == 'ADJ'])
    VERB.append([t.text for t in doc if t.pos_ == 'VERB'])
    
rleaves = rleaves.assign(ADJ=ADJ, VERB=VERB)

# + jupyter={"outputs_hidden": true}
for doc in nlp.pipe(texts, batch_size=50, n_process=3):
    print([t.text for t in doc if t.dep_ == 'nmod'])
# -

rleaves.raw[11]


def visualize(text):
    docs = list(nlp.pipe(text))
    #entities = [doc.ents for doc in docs]
    #options = {"ents": ['ORG']}
    displacy.render(docs, style="dep")#, options=options)


doc = nlp(rleaves.raw[11])
sentence_spans = list(doc.sents)
displacy.render(sentence_spans, style="dep")


def find_days(text):
    matcher = Matcher(nlp.vocab)
    pattern = [{'LIKE_NUM': True},
               {'POS': 'NOUN'}]
    #pattern = [{'DEP': 'ccomp'}]
    matcher.add("test", None, pattern)
    matches = matcher(text)
    print('Total matches found:', len(matches))
    for match_id, start, end in matches:
        span = doc[start:end]
        print(span.text)    


doc = nlp(' '.join(rleaves.raw))

# !jupytext --to py BS-FeatureEngineering.ipynb


