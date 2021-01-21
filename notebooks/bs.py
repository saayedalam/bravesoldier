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
    

def introduction():
    intro = st.beta_expander('Introduction', expanded=True)
    intro.markdown(get_file_content_as_string("instructions.md"))
    image = Image.open('wordcloud_user_leaves.png')
    st.image(image, caption='Word Cloud of r/leaves', use_column_width=True)
    
@st.cache(show_spinner=False)
def get_file_content_as_string(path):
    url = 'https://raw.githubusercontent.com/streamlit/demo-self-driving/master/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")

if __name__ == "__main__":
    main()
    
    
