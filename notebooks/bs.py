import urllib
import textacy.ke
import pandas as pd
import streamlit as st
import plotly.express as px

from PIL import Image
from collections import Counter

def main():
    # Streamlit Layout Configuration
    st.set_page_config(page_title='by Saayed Alam', page_icon=':leaves:')
    st.markdown("# *Data Analysis* of **r/leaves**")
    st.sidebar.markdown("### r/leaves")
    sidebar = st.sidebar.radio("Table Of Content",
                               ("Introduction", "Post", "Authors", "Time"))
    # Sidebar Selection
    if sidebar == "Introduction":
        introduction()
    elif sidebar == "Post":
        posts()

def introduction():
    # Section 1: Introduction
    intro = st.beta_expander('Introduction', expanded=True)
    intro.markdown(get_file_content_as_string("introduction.md"))
    # Section 2: Word Cloud PNG
    image = Image.open('wordcloud_user_leaves.png')
    st.image(image, caption='Word Cloud of r/leaves', use_column_width=True)
    with st.beta_expander("Code"):
        st.code(get_file_content_as_string("rleaves_wc.py"), language='python')
        
def posts():
    # Section 1 : Displays before and after texts
    intro = st.beta_expander('Before & After of Posts Cleanup')
    intro.markdown(''':dart: The truth is cleaning text was the most time consuming and nerve wracking part of this project. 
However, if Data Science can be sexy, so can be cleaning. :smirk::point_down: \n I used [Spacy](https://spacy.io/) and Python 
to transform all the data by lowering, regexing, lemmatizing; and removing stopwords, emojis, and spaces. :tired_face: 
\n ![Alt](https://media.giphy.com/media/26gscNQHswYio5RBu/giphy.gif)''')
    df1 = pd.read_csv('rleaves.csv', encoding='utf-8')
    df1 = df1[['raw', 'time']]
    df2 = pd.read_csv('rleaves_clean.csv', encoding='utf-8')
    col1, col2 = st.beta_columns(2)
    col1.dataframe(df1)
    col2.dataframe(df2)
    with st.beta_expander("Code"):
        st.code(get_file_content_as_string("text_cleanup.py"), language='python')    
    # Section 2 : lets user select most used words
    with st.beta_expander("Most Used Words of r/leaves"):
        st.markdown("Hello")
    number = st.number_input('Select a number to show count', max_value=85229, value=50) # streamlit
    plot(get_top_words(df2['raw'], number), 'Top Words', 'green')
    
    # Section 3: Shows top ranked bigram words using yake algorithm
    st.subheader('Most Important Bigram Word using YAKE! Algorithm')
    st.table(get_top_ranked(df2['raw']))    
    
@st.cache(show_spinner=False)
def get_file_content_as_string(path):
    url = 'https://raw.githubusercontent.com/saayedalam/bravesoldier/main/notebooks/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")

def plot(df, title, color):
    fig = px.bar(df, x=df.iloc[:, 0], y=df.iloc[:, 1], template='plotly_white')
    fig.update_yaxes(visible=False)
    fig.update_xaxes(title_text=title)
    fig.update_traces(marker_color=color)
    st.plotly_chart(fig)

@st.cache(show_spinner=False)
def get_top_words(data, number):
    top_words = Counter(' '.join(data).split()).most_common(int(number))
    top_words = pd.DataFrame(top_words, columns=['word', 'count'])
    return top_words

@st.cache(show_spinner=False)
def get_top_ranked(data):
    en = textacy.load_spacy_lang('en_core_web_sm')
    text = ' '.join(data)
    doc = textacy.make_spacy_doc(text, lang=en)
    df = pd.DataFrame(textacy.ke.yake(doc, ngrams=2, window_size=4, topn=10)).rename(columns={0: "Most Important Keywords", 1: "YAKE! Score"})
    return df
    

if __name__ == "__main__":
    main()
    
    
