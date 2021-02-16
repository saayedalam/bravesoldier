import re
import urllib
import inspect
import textacy.ke
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from textacy import *
from PIL import Image
from collections import Counter
from streamlit_disqus import st_disqus

def main():
    # Streamlit Layout Configuration
    st.set_page_config(page_title='by Saayed Alam', page_icon=':leaves:')
    st.markdown("# *Data Analysis* of **r/leaves**")
    st.sidebar.markdown("### r/leaves")
    sidebar = st.sidebar.radio("Table Of Content",
                               ("Introduction", "Post", "Authors", "Time", "Emotion", "Conclusion"))
    # Load the data
    df = pd.read_csv('rleaves_clean.csv', encoding='utf-8')
    # Sidebar Selection
    if sidebar == "Introduction":
        introduction()
    elif sidebar == "Post":
        posts(df)
    elif sidebar == "Authors":
        authors(df)
    elif sidebar == "Time":
        st.sidebar.info('Please give few seconds for it to load.')
        time(df)
    elif sidebar == "Emotion":
        emotions()
    elif sidebar == "Conclusion":
        st.markdown(''':rewind: I believe there is more to learn from this subreddit. And I plan to explore more using natural language processing and machine learning. Nonetheless, I learned the following so far from r/leaves. :points_down:  
        :star2: Time is felt differently when one is high in contrast to when one is sober. Hence, the importance of time.    
        :star2: The start of a day or the start of a week are most succesfull time for a change.   
        :star2: Change is possible no matter how long it takes.   
        ![next](https://media.giphy.com/media/u3F09PUxoQ11QXdItB/giphy.gif)  
        ''')
        st_disqus("saayedalam")

def introduction():
    # Section 1: Introduction
    intro = st.beta_expander('Introduction', expanded=True)
    intro.markdown(get_file_content_as_string("README.md"))
    # Section 2: Word Cloud PNG
    image = Image.open('wordcloud_user_leaves.png')
    st.image(image, caption='Word Cloud of r/leaves', use_column_width=True)
    with st.beta_expander("Code"):
        st.code(get_file_content_as_string("rleaves_wc.py"), language='python')
        
def posts(data):
    # Section 1 : Displays before and after texts
    intro = st.beta_expander('Before & After of Posts Cleanup', expanded=True)
    intro.markdown(''':dart: The truth is cleaning text was the most time consuming and nerve wracking part of this project. 
    However, if Data Science can be sexy, so can be cleaning. :smirk::point_down: \n I used [Spacy](https://spacy.io/) and Python 
    to transform all the data by lowering, regexing, lemmatizing; and removing stopwords, emojis, and spaces. :tired_face:   
    ![Alt](https://media.giphy.com/media/26gscNQHswYio5RBu/giphy.gif)''')
    df1 = pd.read_csv('rleaves.csv', encoding='utf-8')
    df1 = df1[['raw', 'time']]
    #df2 = pd.read_csv('rleaves_clean.csv', encoding='utf-8')
    col1, col2 = st.beta_columns(2)
    col1.dataframe(df1)
    col2.dataframe(data)
    with st.beta_expander("Code"):
        st.code(get_file_content_as_string("text_cleanup.py"), language='python')    
    # Section 2 : lets user select most used words
    with st.beta_expander("Most Used Words of r/leaves", expanded=True):
        st.markdown('''![time](https://media.giphy.com/media/9u514UZd57mRhnBCEk/giphy.gif)   
        :1234: If you are subscribed to r/leaves, these top words are not that surprising. The topic of discussion
        weed is the most used word followed by several mentions of time. In my opinion, people mourn the loss of time due to intoxication. 
        To see more words and the frequency of their appearance. :point_down:   
        :helicopter: P.S. You can hover over a bar to see more details.''')
    number = st.number_input('Select A Number:', max_value=85229, value=50) # streamlit
    plot(get_top_words(data['raw'], number), 'Top Words', 'green')
    with st.beta_expander("Code"):
        st.code(get_file_content_as_string("rleaves_wc.py"), language='python') 
    # Section 3: Shows top ranked bigram words using yake algorithm
    with st.beta_expander('Most Important Bigram Word using YAKE! Algorithm', expanded=True):
        st.markdown('''![wtf](https://media.giphy.com/media/pPhyAv5t9V8djyRFJH/giphy.gif)   
        What is a YAKE! Algorithm you make ask.    
        *YAKE! is a light-weight unsupervised automatic keyword extraction method which rests on text statistical 
        features extracted from single documents to select the most important keywords of a text.*   
        :hourglass_flowing_sand: As we can see from the table, time and marijuana are inseparable in r/leaves. :leaves:''')
    st.table(get_top_ranked(data['raw']))    
    with st.beta_expander("Code"):
        st.code(inspect.getsource(get_top_ranked), language='python')  
        
def authors(data):
    # Section 1: Some stats about the posters frequency of posts by day
    with st.beta_expander("Frequency of Posts by Day", expanded=True):
        st.markdown(':calendar: **Mondays** saw more new posts than any other days. In my opinion, it is because with the start of a new week, users wanted to have a new start. Whereas, **Fridays** and **Saturdays** were harder because people tend to hang out those days.')
    with st.echo(code_location='below'):
        fig1 = pd.DataFrame(data['day'].value_counts()).reset_index()
        fig1.rename(columns={"index": "day", "day": "count"})
        plot(fig1, 'Number of Posts per Day', 'plum')
    # Section 2: same as above buy per hours
    with st.beta_expander("Frequency of Posts by Day", expanded=True):
        st.markdown(':calendar: Late nights saw very few posts. For instance, from **12 AM** to **4 AM** we see the least amount of posts. In my opinion, it is harder to have stong will power at that time of the hour. Whereas early mornings saw most posts, i.e. **7 AM** - **1 PM**, than others. I believe it is due to us having a more clearer mind after a good night of sleep and being able to make more rationale decisions.')
    with st.echo(code_location='below'):
        fig1 = pd.DataFrame(data['hour'].value_counts()).reset_index()
        fig1.rename(columns={"index": "time", "hour": "count"})
        plot(fig1, 'Number of Posts per Hour', 'plum')
    # Section 3: some anonymous stats about the authors
    with st.beta_expander("Frequency of Posts by Authors", expanded=True):
        st.markdown(''':lock: Due to privacy concerns, I did not use redittors' data for analyzation. However, I pulled the following statistics.   
        ![stat](https://media.giphy.com/media/CtqI1GmvT0YVO/giphy.gif)  
        :chart_with_upwards_trend: **805** authors appeared in the top 1000 posts of all time.  
        :chart_with_upwards_trend: **81.23%** of those posts are by unique authors.  
        :chart_with_upwards_trend: **77 authors** have more than one post.''')
    with st.beta_expander("Code"):
        st.code("""round(rleaves['author'].nunique()/rleaves.shape[0]*100, 2)
rleaves['author'].value_counts().loc[rleaves['author'].value_counts().values > 1].shape[0]""")
        
def time(data):
    # Section 1: This section is for plotting 'day' mentions in subreddit
    with st.beta_expander("Frequency of Reported Duration by Day", expanded=True):
        st.markdown('''![day](https://media.giphy.com/media/CxTLB1wMzPviR8fW4S/giphy.gif)  
        If you read any of the post, you will notice most of the posts start something like this, "*Day 1, Today I stop smoking.*" or "*Day 30, I feel like a new person*". My goal was to extract such reported days.  
        :chart: The plot tells us two things:  
        :small_red_triangle: First **seven days** are reported the most. After which the number drops. It could be because the author stopped posting or the author returned to smoking.  
        :small_red_triangle: Milestone numbers are reported more frequently such as **Day 7**, **10** and **30**.''')
    plot(get_day(data['raw'])[0], 'Day', 'salmon')    
    # Section 2: This section is for plotting 'week' mentions in subreddit
    with st.beta_expander("Frequency of Reported Duration by Week", expanded=True):
        st.markdown(''':chart: The graph is similar to days as we can infer two things from it:  
        :small_red_triangle: **First week** is reported more than half of the mentions; followed by **Week 2**, **3**, and **4**.  
        :small_red_triangle: Only first week seems to be the milestone among the authors.''')        
    plot(get_week(data['raw']), 'Week', 'salmon')    
    # Section 3: This section is for plotting 'month' mentions in subreddit
    with st.beta_expander("Frequency of Reported Duration by Month", expanded=True):
        st.markdown('''![month](https://media.giphy.com/media/d1GpZTVp2eV7gQk8/giphy.gif)  
        :small_red_triangle: First thing we notice, months are reported more than the weeks.  
        :small_red_triangle: First six months are significant milestones. After which the number drops off drastically.   
        :small_red_triangle: **Month 6** is mentioned more than **Month 1**. It could be because authors counted their abstinence by day.  ''')        
    plot(get_month(data['raw']), 'Month', 'salmon')    
    # # Section 4: This section is for plotting 'day' mentions in subreddit
    with st.beta_expander("Frequency of Reported Duration by Year* ", expanded=True):
        st.markdown(''':keycap_star: The number of *year* extracted is ambiguous.  
        :small_red_triangle: **Year** mentioned could be the age of the author.  
        :small_red_triangle:  Or, number of years elapsed since the author has smoked.  
        ![year](https://media.giphy.com/media/d3yxg15kJppJilnW/giphy.gif)''')        
    plot(get_year(data['raw']), 'Year', 'salmon')
    # Section 5: code for all the plots
    with st.beta_expander("Code"):
        st.code(inspect.getsource(plot), language='python')     
        
def emotions():
    #
    with st.beta_expander("Emotion Detection", expanded=True):
        st.markdown('''Hello''')  
    df = pd.read_csv('rleaves_emotion.csv', encoding='utf-8')
    emote = pd.DataFrame(df['emotion'].value_counts().reset_index())
    plot(emote, 'Emotions', 'teal')
    #
    with st.beta_expander("Code"):
        st.code(get_file_content_as_string("emotion_detection.py"), language='python')
    #
    color = st.radio('Select An Emotion', 
                             options=['Joy', 'Sadness', 'Anger', 'Fear', 'Surprise', 'Love'])
    st.table(df['post'][df['emotion'] == color.lower()].head())    
        
@st.cache(show_spinner=False)
def get_file_content_as_string(path):
    url = 'https://raw.githubusercontent.com/saayedalam/bravesoldier/main/' + path
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

@st.cache()
def day_to_week(df):
    return df.groupby(pd.cut(df.iloc[:, 0], np.arange(0, df.iloc[:, 0].max(), 7))).sum()

@st.cache()
def get_day(series):
    day = list(series.str.findall(r'\d+\s*day[s\s]|\s*day\s*\d+'))
    day = [int(item) for item in re.findall(r'\d+', str(day))]
    day = pd.DataFrame([[x, day.count(x)] for x in set(day)]).rename(columns={0:'day', 1:'count'})
    day_week = day.copy()
    day = day.loc[(day['day'] > 0) & (day['day'] < 31)]
    day['day'] = 'Day ' + day['day'].astype(str)
    day.sort_values(by='count', ascending=False, inplace=True) 
    return day, day_week

@st.cache()
def get_week(series):
    #global day_week
    week = list(series.str.findall(r'\d+\s*week[s\s]|\s*week\s*\d+'))
    week = [int(item) for item in re.findall(r'\d+', str(week))]
    week = pd.DataFrame.from_dict(Counter(week), orient='index').rename(columns={0:'w_count'}).sort_index(ascending=True)
    week = week[week.index < 35]
    week.index = 'Week ' + week.index.astype(str)
    day_week = day_to_week(get_day(series)[1])
    day_week = day_week.drop(['day'], axis=1).reset_index(drop=True)
    day_week = day_week[day_week['count'] > 1]
    day_week.index = ['Week %s' %i for i in range(1, len(day_week) + 1)]
    week = pd.concat([day_week, week], axis=1).fillna(0).reset_index()
    week[['count','w_count']] = week[['count','w_count']].astype(int)
    week['count'] = week['count'] + week['w_count']
    week.sort_values(by='count', ascending=False, inplace=True)
    week.drop(columns=['w_count'], inplace=True)
    return week   

@st.cache()
def get_month(series):
    month = list(series.str.findall(r'\d+\s*month[s\s]|\s*month\s*\d+'))
    month = [int(item) for item in re.findall(r'\d+', str(month))]
    month = pd.DataFrame([[x, month.count(x)] for x in set(month)]).rename(columns={0:'month', 1:'count'}).sort_values(by='count', ascending=False)
    month = month[month['month'] < 200]
    month['month'] = 'Month ' + month['month'].astype(str)
    return month

@st.cache()
def get_year(series):
    year = list(series.str.findall('\d+\s*ye?a?r[s\s]*')) 
    year = [int(item) for item in re.findall(r'\d+', str(year))]
    year = pd.DataFrame([[x, year.count(x)] for x in set(year)]).rename(columns={0:'year', 1:'count'}).sort_values(by='count', ascending=False)
    year = year.loc[(year['year'] < 50) & (year['year'] > 0)]
    year['year'] = 'Year ' + year['year'].astype(str)
    return year

if __name__ == "__main__":
    main()
    
    
