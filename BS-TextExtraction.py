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
import os
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
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
from spacy import displacy
from spacy.matcher import Matcher


# + jupyter={"source_hidden": true}
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
                       'gon', 'pas', 'ill', 'youve', 'wont', 'havent', 
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

# +
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

# +
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

# +
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

# +
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
plt.text(50, 50, 'year: a) number of years smoked OR b) age', style='normal' , fontsize=12, bbox=dict(facecolor='blue', alpha=0.2))
plt.xticks(rotation=0)
plt.show()

# +
# Lemmatize Words
from nltk.corpus import stopwords
lemmatizer = WordNetLemmatizer()
stopwords = set(stopwords.words('english')) 

def preprocessing(text):
    text = word_tokenize(text)
    text = [w for w in text if not w in stopwords] 
    text = [lemmatizer.lemmatize(w) for w in text]
    text = ' '.join(text)
    text = re.sub('\sle\s', ' less ', str(text))
    return text

# Applying lemmatize function
rleaves = pd.DataFrame(rleaves['raw'].apply(preprocessing))

# +
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
plt.xlabel('Number of times a word appears')
plt.ylabel('Words')
plt.title('Top Words in r/leaves subreddit')
plt.show()

# Word Embedding
corpus = rleaves['raw'].str.replace(r'\d+', '').apply(word_tokenize).values.tolist()
model_cbow = Word2Vec(corpus, min_count=9, window=3, sg=0, seed=1)
print(tabulate(model_cbow.most_similar('weed'), tablefmt='grid'))
#model_cbow.save('model_cbow.bin')

# +
# Load a spacy model, which will be used for all further processing.
en = textacy.load_spacy_lang('en_core_web_sm')
text = ' '.join(rleaves['raw'])

#convert the text into a spacy document.
doc = textacy.make_spacy_doc(text, lang=en)
# -

textacy.ke.textrank(doc, window_size=10, edge_weighting='count', position_bias=True, topn=10)

print(tabulate(textacy.ke.yake(doc, ngrams=2, window_size=4, topn=10), tablefmt='grid'))

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)
doc = nlp(' '.join(rleaves.raw))
pattern = [{'DEP': 'acl'}, {'POS': 'NOUN'}]
matcher.add("numeric modifier", None, pattern)
matches = matcher(doc)
for match_id, start, end in matches:
    # Get the matched span
    matched_span = doc[start:end]
    print(matched_span.text)

rr.raw[0]



rleaves[rleaves.raw.str.contains(r'\d+')]

# + jupyter={"outputs_hidden": true}
# !jupytext --to py BS-TextExtraction.ipynb
