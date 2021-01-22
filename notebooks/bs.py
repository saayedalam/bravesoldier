import urllib
import pandas as pd
import streamlit as st
from PIL import Image

def main():
    # layout configuration
    st.set_page_config(page_title='by Saayed Alam', page_icon=':leaves:')
    st.markdown("# *Data Analysis* of **r/leaves**")
    st.sidebar.markdown("### r/leaves")
    sidebar = st.sidebar.radio("Table Of Content",
                               ("Introduction", "Post", "Authors", "Time"))
    
    if sidebar == "Introduction":
        introduction()
    elif sidebar == "Post":
        posts()

def introduction():
    intro = st.beta_expander('Introduction', expanded=True)
    intro.markdown(get_file_content_as_string("introduction.md"))
    image = Image.open('wordcloud_user_leaves.png')
    st.image(image, caption='Word Cloud of r/leaves', use_column_width=True)
    with st.beta_expander("Code"):
        st.code('x=x', language='python')
        
def posts():
    df1 = pd.read_csv('rleaves.csv', encoding='utf-8')
    df1 = df1[['raw', 'time']]
    df2 = pd.read_csv('rleaves_clean.csv', encoding='utf-8')
    col1, col2 = st.beta_columns(2)
    col1.subheader('Before Cleanup')
    col1.dataframe(df1)
    col2.subheader('After Cleanup')
    col2.dataframe(df2)
    with st.beta_expander("Code"):
        st.code(get_file_content_as_string("text_cleanup.py"), language='python')
    
    # Most common words
    st.subheader("Most Used Words of r/leaves")
    #number = st.number_input('Select a number to show count', max_value=85229, value=50) # streamlit
    #top_words = Counter(' '.join(rleaves['raw']).split()).most_common(int(number))
    #top_words = pd.DataFrame(top_words, columns=['word', 'count'])
    #fig = px.bar(top_words, x='count', y='word', orientation='h', template='plotly_white')
    #fig.update_yaxes(visible=False, categoryorder='total ascending')
    #fig.update_layout(hovermode="x unified")
    #fig.update_traces(marker_color='green')
    #st.plotly_chart(fig, use_container_width=True)
    
@st.cache(show_spinner=False)
def get_file_content_as_string(path):
    url = 'https://raw.githubusercontent.com/saayedalam/bravesoldier/main/notebooks/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")

if __name__ == "__main__":
    main()
    
    
