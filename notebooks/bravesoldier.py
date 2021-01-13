# required libraries
import re
import spacy
import string
import warnings
import textacy.ke
import numpy as np
import pandas as pd
import datetime as dt
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
plt.set_loglevel('WARNING')
sp = spacy.load('en_core_web_sm')

from PIL import Image
from spacy import displacy
from tabulate import tabulate
from textblob import TextBlob
from string import punctuation
from collections import Counter
from nltk.corpus import stopwords
from spacy.matcher import Matcher
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud, STOPWORDS
from gensim.models import Word2Vec, KeyedVectors
from nltk.tokenize import sent_tokenize, word_tokenize

# sidebar
st.sidebar.markdown("### r/leaves")
sidebar = st.sidebar.radio("Table Of Content", ("Introduction", "Post", "Authors", "Time"))

# title
st.markdown("# *Data Analysis* of **r/leaves**")

if sidebar == "Introduction":
    intro = st.beta_expander('Introduction', expanded=True)
    intro.write("""*r/leaves* is a support and recovery community for practical discussions about how to quit pot, weed, cannabis, edibles, BHO, shatter, or whatever THC-related product, and support in staying stopped.""")
    image = Image.open('wordcloud_user_leaves.png')
    st.image(image, caption='Word Cloud of r/leaves', use_column_width=True)
    with st.beta_expander("Code"):
        code = '''# Word Cloud
wordcloud_text = ' '.join(df['raw'].tolist())

def plot_cloud(wordcloud):
    plt.figure(figsize=(40, 30))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    
# WordCloud with Mask
mask = np.array(Image.open('marijuana.png'))
wordcloud_user = WordCloud(width=300, height=200, background_color='rgba(255, 255, 255, 0)', mode='RGBA', colormap='tab10', collocations=False, mask=mask, include_numbers=True)
wordcloud_user.generate(wordcloud_text)
#wordcloud_user.to_file("wordcloud_user_leaves.png") # saves the image
plot_cloud(wordcloud_user)'''
        st.code(code, language='python')
                
        
# Load the local files
rleaves = pd.read_csv('rleaves.csv', encoding='utf-8')
df1 = rleaves[['raw', 'time']]

# a custom function for complete text cleanup
@st.cache
def cleanup1(text):
    text = text.lower() # lowers the corpus
    text = re.sub('http\S+', ' ', str(text)) # removes any url
    text = re.sub('n\'t\s', ' not ', str(text))
    text = re.sub('-(?<!\d)', ' ', str(text)) # removing hyphens from numbers
    text = sp(text) # apply spacy model
    text = [w.text for w in text if not w.is_stop] # tokenize and remove stop words
    text = sp(' '.join(text)) # join words and apply spacy model again 
    text = [w.lemma_ for w in text] # lemmatizes the words
    stopwords_extra = ['im', 'na', 'u', 'ill', '10184285', '179180', 'as', 'oh', 'av', 'wo', 'nt', 'p', 'm', 'ta', '10000', '6000']
    text = [word for word in text if not word in stopwords_extra] # remove additional unnecessary words
    text = ' '.join(text)  # join the words back together  
    text = text.translate(str.maketrans('', '', string.punctuation)) # removes all punctuation
    text = re.sub('[^\w\s,]', ' ', str(text)) # removes emoticon and other non characters
    text = re.sub('x200b', ' ', str(text)) # removing zero-width space characters
    text = re.sub(' cannabi ', ' cannabis ', str(text))
    return ' '.join([token for token in text.split()]) # removes trailing whitespaces

# a custom function to change UTC time and split days and hours
@st.cache
def change_time(utc):
    day = dt.datetime.fromtimestamp(utc).strftime('%A')
    hour = dt.datetime.fromtimestamp(utc).strftime('%I %p')
    return pd.Series([day, hour])

# apply preprocessing function
rleaves[['day', 'hour']] = rleaves['time'].apply(change_time)
rleaves['raw'] = pd.DataFrame(rleaves['raw'].apply(cleanup1))
df2 = rleaves[['raw', 'day', 'hour']]

# display the data
if sidebar == "Post":
    col1, col2 = st.beta_columns(2)
    col1.subheader('Before Cleanup')
    col1.dataframe(df1)
    col2.subheader('After Cleanup')
    col2.dataframe(df2)
    # Most common words
    st.subheader("Most Used Words of r/leaves")
    number = st.number_input('Select a number to show count', max_value=85229, value=50) # streamlit
    top_words = Counter(' '.join(rleaves['raw']).split()).most_common(int(number))
    top_words = pd.DataFrame(top_words, columns=['word', 'count'])
    fig = px.bar(top_words, x='count', y='word', orientation='h', template='plotly_white')
    fig.update_yaxes(visible=False, categoryorder='total ascending')
    fig.update_layout(hovermode="x unified")
    fig.update_traces(marker_color='green')
    st.plotly_chart(fig, use_container_width=True)
    
    # Load a spacy model, which will be used for all further processing.
    en = textacy.load_spacy_lang('en_core_web_sm')
    text = ' '.join(rleaves['raw'])
    
    #convert the text into a spacy document.
    doc = textacy.make_spacy_doc(text, lang=en)
    df = pd.DataFrame(textacy.ke.yake(doc, ngrams=2, window_size=4, topn=10)).rename(columns={0: "Most Important Keywords", 1: "YAKE! Score"})
    st.subheader('Most Important Bigram Word using YAKE! Algorithm')
    st.table(df)    

if sidebar == "Authors":
    # authors about day
    with st.beta_expander("What day of the week had more posts than the others?", expanded=True):
        st.markdown('Monday saw more new posts than any other days. In my opinion, it is because with the start of a new week, users wanted to have a new start')
        
    with st.echo(code_location='below'):
        fig1 = pd.DataFrame(rleaves['day'].value_counts()).reset_index().rename(columns={"index": "day", "day": "count"})
        fig = px.bar(fig1, x='day', y='count', template='plotly_white')
        fig.update_yaxes(visible=False)
        fig.update_xaxes(title_text='Day')
        fig.update_traces(marker_color='plum')
        st.plotly_chart(fig)
        
    
    with st.beta_expander("What time of the day had more posts than the others?", expanded=True):
        st.markdown('Late nights saw very few posts. For instance, from 12 AM to 4 AM we see the least amount of posts. In my opinion, it is harder to have stong will power at that time of the hour. That is why we see more posts during mid day.')
        
    with st.echo(code_location='below'):
        fig2 = pd.DataFrame(rleaves['hour'].value_counts()).reset_index().rename(columns={'index': 'time', 'hour': 'count'})
        fig = px.bar(fig2, x='time', y='count', template='plotly_white')
        fig.update_yaxes(visible=False)
        fig.update_xaxes(title_text='Time')
        fig.update_traces(marker_color='plum')
        st.plotly_chart(fig)
        
    with st.beta_expander(" What do we know about the authors of the posts?", expanded=True):
        st.markdown('''> **81.23%** of the posts are by unique authors.
        >
        > **77 authors** have more than one post.''')
        
    with st.beta_expander("Code"):
        st.code("""round(rleaves['author'].nunique()/rleaves.shape[0]*100, 2)
rleaves['author'].value_counts().loc[rleaves['author'].value_counts().values > 1].shape[0]""")
    
# Time Sidebar    
if sidebar == "Time":
    
    def plot(df, title):
        fig = px.bar(df, x=df.iloc[:, 0], y=df.iloc[:, 1], template='plotly_white')
        fig.update_yaxes(visible=False)
        fig.update_xaxes(title_text=title)
        fig.update_traces(marker_color='salmon')
        st.plotly_chart(fig)
    
    st.cache()
    def day_to_week(df):
        return df.groupby(pd.cut(df.iloc[:, 0], np.arange(0, df.iloc[:, 0].max(), 7))).sum()
    
    st.cache()
    def get_month(series):
        month = list(series.str.findall(r'\d+\s*month[s\s]|\s*month\s*\d+'))
        month = [int(item) for item in re.findall(r'\d+', str(month))]
        month = pd.DataFrame([[x, month.count(x)] for x in set(month)]).rename(columns={0:'month', 1:'count'}).sort_values(by='count', ascending=False)
        month = month[month['month'] < 200]
        month['month'] = 'Month ' + month['month'].astype(str)
        return month

    st.cache()
    def get_year(series):
        year = list(series.str.findall('\d+\s*ye?a?r[s\s]*')) 
        year = [int(item) for item in re.findall(r'\d+', str(year))]
        year = pd.DataFrame([[x, year.count(x)] for x in set(year)]).rename(columns={0:'year', 1:'count'}).sort_values(by='count', ascending=False)
        year = year.loc[(year['year'] < 50) & (year['year'] > 0)]
        year['year'] = 'Year ' + year['year'].astype(str)
        return year
    
    
    day = list(rleaves.raw.str.findall(r'\d+\s*day[s\s]|\s*day\s*\d+'))
    day = [int(item) for item in re.findall(r'\d+', str(day))]
    day = pd.DataFrame([[x, day.count(x)] for x in set(day)]).rename(columns={0:'day', 1:'count'})
    day_week = day.copy()
    day = day.loc[(day['day'] > 0) & (day['day'] < 31)]
    day['day'] = 'Day ' + day['day'].astype(str)
    day.sort_values(by='count', ascending=False, inplace=True) 
     
    day_week = day_to_week(day_week)
    day_week = day_week.drop(['day'], axis=1).reset_index(drop=True)
    day_week = day_week[day_week['count'] > 1]
    day_week.index = ['Week %s' %i for i in range(1, len(day_week) + 1)]
    week = list(rleaves.raw.str.findall(r'\d+\s*week[s\s]|\s*week\s*\d+'))
    week = [int(item) for item in re.findall(r'\d+', str(week))]
    week = pd.DataFrame.from_dict(Counter(week), orient='index').rename(columns={0:'w_count'}).sort_index(ascending=True)
    week = week[week.index < 35]
    week.index = 'Week ' + week.index.astype(str)
    week = pd.concat([day_week, week], axis=1).fillna(0).reset_index()
    week[['count','w_count']] = week[['count','w_count']].astype(int)
    week['total_count'] = week['count'] + week['w_count']
    week.sort_values(by='total_count', ascending=False, inplace=True)
    
  
    # This section is for plotting 'day' mentions in subreddit
    with st.beta_expander("Which day during an author's journey had the most post?", expanded=True):
        st.markdown('XXX')
    plot(day, 'Day')
    
    # This section is for plotting 'week' mentions in subreddit
    with st.beta_expander("Which week during an author's journey had the most post?", expanded=True):
        st.markdown('XXX')        
    fig = px.bar(week, x='index', y='total_count', template='plotly_white')
    fig.update_yaxes(visible=False)
    fig.update_xaxes(title_text='Week')
    fig.update_traces(marker_color='salmon')
    st.plotly_chart(fig)
    
    # This section is for plotting 'month' mentions in subreddit
    with st.beta_expander("Which month during an author's journey had the most post?", expanded=True):
        st.markdown('XXX')        
    plot(get_month(rleaves['raw']), 'Month')
    
    # # This section is for plotting 'day' mentions in subreddit
    with st.beta_expander("Which year during an author's journey had the most post?", expanded=True):
        st.markdown('XXX')        
    plot(get_year(rleaves['raw']), 'Year')
