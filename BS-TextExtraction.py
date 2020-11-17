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
import praw
import re
import string
import spacy
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
# %matplotlib inline
warnings.filterwarnings('ignore')

from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from spellchecker import SpellChecker
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
from spacy import displacy


# + jupyter={"outputs_hidden": true, "source_hidden": true}
## Reddit API Credentials
#reddit = praw.Reddit(client_id='7_PY9asBHeVJxw',
#                     client_secret='KL01wgTYZqwEDdPH-R8vNBqFYe4',
#                     password='9S2a8a7hcr!',
#                     user_agent='bravesoldier by /u/saayed',
#                     username='saayed')
#
## Pull the subreddit of 
#subreddit = reddit.subreddit('leaves')
#
## Pulling top 1000 posts of leaves subreddit
#leaves_subreddit = reddit.subreddit('leaves').top(limit=1000)
#
## Detail Information on the subreddit
##print(subreddit.display_name, subreddit.title, subreddit.description, sep="\n")
#
## Create an empty dictionary to save data
#dict = {'title': [],
#        'body': [],
#       }
#
## Storing the data in the empty dictionary
#for submission in leaves_subreddit:
#    dict['title'].append(submission.title)
#    dict['body'].append(submission.selftext)
#
## Convert the data to pandas dataframe and apply date function
#df = pd.DataFrame(dict)
#df['raw'] = df['title'] + ' ' + df['body']
#df.drop(['title', 'body'], axis=1, inplace=True)
#
## Print the first 5 rows of the data
##df.head()
#
## Save it as CSV
#df.to_csv('rleaves.csv', index=False)

# +
# Cleaning up the corpus
def cleanup(text):
    
    # lowercase the corpus 
    text = text.lower()
    # removing apostrophes
    text = re.sub("'s", ' ', str(text))
    # removing hyphens from numbers
    text = re.sub('-(?<!\d)', ' ', str(text))
    # removing punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # removing emoticons
    text = re.sub('[^\w\s,]', ' ', str(text))
    # removing zero-width space characters
    text = re.sub('x200b', ' ', str(text))
    # removing trailing whitespaces
    text = ' '.join([token for token in text.split()])
    # word tokenization
    text = word_tokenize(text)
    # additional removal of unnecessary words
    stopwords_extra = ['im', 'ive', 'dont', 'didnt', 'doesnt', 'isnt', 
                       'couldnt', 'na', 'youre', 'cant', 'u', 'id', 'wasnt', 
                       'le', 'gon', 'pas', 'ill', 'youve', 'wont', 'havent', 
                       'wouldnt', '10184285', '179180', 'arent', 'youll', 'as', 
                       'oh', 'wan', 'av', 'p', 'ta', '10000', '6000']
    text = [word for word in text if not word in stopwords_extra]
    # join the words
    text = ' '.join(text)

    return text

# Load the local files
rleaves = pd.read_csv('rleaves.csv', encoding='utf-8')

# apply preprocessing function
rleaves = pd.DataFrame(rleaves['raw'].apply(cleanup))

# + jupyter={"outputs_hidden": true}
# DAY
day = list(rleaves.raw.str.findall(r'\d+\s*day[s\s]|\s*day\s*\d+'))
day = [int(item) for item in re.findall(r'\d+', str(day))]
day = pd.DataFrame.from_dict(Counter(day), orient='index').reset_index().rename(columns={'index':'day', 0:'count'})
# Bin the days in to 7 day increment
day = day.groupby(pd.cut(day['day'], np.arange(0, day['day'].max(), 7))).sum()
day = day.drop(['day'], axis=1).reset_index(drop=True)
day.index = ['Week %s' %i for i in range(1, len(day) + 1)]
day = day[day['count'] > 1]

# Visualizing the days
plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(25, 10))
plt.bar(day.index, day['count'], color='salmon')
plt.xlabel("Period of Time")
plt.ylabel("Number of Appearance")
plt.xticks(rotation=90)
plt.title("Number of times 'Day' appeared in the r/leaves")
plt.show()

# + jupyter={"outputs_hidden": true}
# week
week = list(rleaves.raw.str.findall(r'\d+\s*week[s\s]|\s*week\s*\d+'))
week = [int(item) for item in re.findall(r'\d+', str(week))]
week = pd.DataFrame.from_dict(Counter(week), orient='index').rename(columns={0:'count'}).sort_index(ascending=True)
week.index = 'Week ' + week.index.astype(str)

# Visualizing the months
plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(20, 10))
plt.bar(week.index, week['count'], color='salmon')
plt.xlabel('Period of Time')
plt.ylabel('Number of Appearance')
plt.title('Number of times "Week" appeared in the r/leaves')
plt.xticks(rotation=0)
plt.show()

# + jupyter={"outputs_hidden": true}
# MONTH
month = list(rleaves.raw.str.findall(r'\d+\s*month[s\s]|\s*month\s*\d+'))
month = [int(item) for item in re.findall(r'\d+', str(month))]
month = pd.DataFrame.from_dict(Counter(month), orient='index').rename(columns={0:'count'}).sort_index(ascending=True)

# Visualizing the months
plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(20, 10))
plt.bar(month.index, month['count'], data=month, color='salmon')
plt.xlabel("Period of Time")
plt.ylabel("Number of Appearance")
plt.title("Number of times 'Month' appeared in the r/leaves")
plt.xticks(rotation=0)
plt.show()

# + jupyter={"outputs_hidden": true}
# YEAR
year = list(rleaves.raw.str.findall(r'\d+\s*ye?a?r[s\s]*')) 
year = [int(item) for item in re.findall(r'\d+', str(year))]
year = pd.DataFrame.from_dict(Counter(year), orient='index').rename(columns={0:'count'}).sort_index(ascending=True)

# Visualizing the years
plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(20, 10))
plt.bar(year.index, year['count'], color='salmon')
plt.xlabel('Period of Time')
plt.ylabel('Number of Appearance')
plt.title('Number of times "Year" appeared in the r/leaves')
plt.text(28, 28, 'Year: a) number of years smoked OR b) age', style='normal')
plt.xticks(rotation=0)
plt.show()
# -

# Lemmatize Words
lemmatizer = WordNetLemmatizer()
stopwords = set(stopwords.words('english')) 


def preprocessing(text):
    text = word_tokenize(text)
    text = [w for w in text if not w in stopwords] 
    text = [lemmatizer.lemmatize(w) for w in text]
    #text = re.sub('\sle\s', 'less', str(text))
    text = ' '.join(text)
    return text


# Applying lemmatize function
rleaves = pd.DataFrame(rleaves['raw'].apply(preprocessing))

# + jupyter={"outputs_hidden": true}
# WordCloud
wordcloud_text = ' '.join(rleaves['raw'].tolist())

def plot_cloud(wordcloud):
    
    plt.figure(figsize=(40, 30))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

    
# WordCloud with Mask
mask = np.array(Image.open('user.png'))
wordcloud_user = WordCloud(width=3000, height=2000, random_state=1, background_color='rgba(255, 255, 255, 0)', mode='RGBA', colormap='tab10', collocations=False, mask=mask).generate(wordcloud_text)
    
#wordcloud_user.to_file("wordcloud_user_leaves.png")
plot_cloud(wordcloud_user)
# -

# Most common words
top_words = Counter(' '.join(rleaves['raw']).split()).most_common(50)
top_words = pd.DataFrame(top_words, columns=['word', 'count']).set_index('word').sort_values(by='count', ascending=True)
plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(20, 30))
plt.barh(top_words.index, top_words['count'], color='salmon')
plt.xlabel('N')
plt.ylabel("Number of Appearance")
plt.title("Number of times 'Month' appeared in the r/leaves")
plt.show()

# + jupyter={"outputs_hidden": true}
# !jupytext --to py BS-TextExtraction.ipynb

# + jupyter={"outputs_hidden": true} endofcell="--"
# # + jupyter={"outputs_hidden": true}
# Word Embedding
corpus = rleaves['raw'].str.replace(r'\d+', '').apply(word_tokenize).values.tolist()
model_cbow = Word2Vec(corpus, min_count=9, window=3, sg=0, seed=1)
model_skipgram = Word2Vec(corpus, min_count=9, window=3, sg=1, seed=1)

print(model_cbow.most_similar('day'))
print(model_skipgram.most_similar('day'))
#model_cbow.save('model_cbow.bin')
#new_model_cbow = Word2Vec.load('model_cbow.bin')

# # +
# TFIDF
#tfidf = TfidfVectorizer()
#bow_rep_tfidf = tfidf.fit_transform(processed_docs)
#
##IDF for all words in the vocabulary
#print("IDF for all words in the vocabulary\n", tfidf.idf_)
#print("_"*10)
#
##All words in the vocabulary
#print("All words in the vocabulary\n", tfidf.get_feature_names())
#print("_"*10)
#
##TFIDF representation of all documents in our corpus
#print("TFIDF representation of all documents in our corpus\n", bow_rep_tfidf.toarray())
#print("_"*10)

# # + jupyter={"outputs_hidden": true}
# Word Embedding
corpus = processed_docs.str.replace(r'\d+', '').apply(word_tokenize).values.tolist()
model_cbow = Word2Vec(corpus, min_count=9, window=3, sg=0, seed=1)
model_skipgram = Word2Vec(corpus, min_count=9, window=3, sg=1, seed=1)

print(model_cbow.most_similar('age'))
print(model_skipgram.most_similar('age'))
#model_cbow.save('model_cbow.bin')
#new_model_cbow = Word2Vec.load('model_cbow.bin')

# # +
#VISUALIZATION

#from gensim.models import Word2Vec, KeyedVectors
#import warnings
#warnings.filterwarnings('ignore')
#
#import numpy as np
#import matplotlib.pyplot as plt
#from sklearn.manifold import TSNE
#
#model = KeyedVectors.load('model_cbow.bin')
#
#words_vocab = list(model.wv.vocab)
#print("Size of Vocabulary: ", len(words_vocab))
#print("="*30)
#print("Few Words in Vocabulary", words_vocab[:50])
# -

import spacy
import textacy.ke
from textacy import *

en = textacy.load_spacy_lang('en_core_web_sm')

# # +
text = ' '.join(rleaves['raw'])

#with open('rleaves.txt', 'w') as output:
#    output.write(text)
# -

doc = textacy.make_spacy_doc(text, lang=en)

textacy.ke.textrank(doc, topn=5)

textacy.ke.textrank(doc, topn=10)

print('Textrank output: ', [kps for kps, weights in textacy.ke.textrank(doc, normalize='lemma', topn=10)])

print('Textrank output: ', [kps for kps, weights in textacy.ke.textrank(doc, normalize='lemma', topn=10)])

# # + active=""
# SPACY CHAPTER 1
#
# from spacy.lang.en import English
#
# nlp = English()
#
# doc = nlp(rleaves['raw'][3])
#
# token = doc[1:3]
#
# print(token.text)
#
# # Process the text
# doc = nlp(' '.join(rleaves['raw']))
#
# # Iterate over the tokens in the doc
# for token in doc:
#     # Check if the token resembles a number
#     if token.like_num:
#         # Get the next token in the document
#         next_token = doc[token.i + 1]
#         # Check if the next token's text equals "%"
#         if next_token.text == "year":
#             print("Percentage found:", token.text)
#
# nlp = spacy.load('en_core_web_sm')
# doc = nlp(rleaves['raw'][1])
#
# for token in doc:
#     print(token.text, token.pos_, token.dep_, token.head.text)
#     
# for ent in doc.ents:
#     print(ent.text, ent.label_)
#
# from spacy.matcher import Matcher
#
# nlp = spacy.load('en_core_web_sm')
# matcher = Matcher(nlp.vocab)
#
# # pattern
# pattern = [{'POS': 'ADJ'}, {'POS': 'NOUN'}, {'POS': 'NOUN', 'OP': '?'}]
# matcher.add('DAY_PATTERN', None, pattern)
#
# doc = nlp(' '.join(rleaves['raw']))
#
# matches = matcher(doc)
#
# #print('Matches:', [doc[start:end].text for match_id, start, end in matches])
#
# for match_id, start, end in matches:
#     print('Match found:', doc[start:end].text)
#
# nlp = spacy.load('en_core_web_sm')
# doc = nlp(rleaves['raw'][2])
#
# for token in doc:
#     # Get the token text, part-of-speech tag and dependency label
#     token_text = token.text
#     token_pos = token.pos_
#     token_dep = token.dep_
#     # This is for formatting only
#     print(f"{token_text:<12}{token_pos:<10}{token_dep:<10}")
#

# # +
#print('SGRank output: ', [kps for kps, weights in textacy.ke.sgrank(doc, topn=10)])

# # +
#terms = set([term for term, weight in textacy.ke.sgrank(doc)])
#print(textacy.ke.utils.aggregate_term_variants(terms))
# --
