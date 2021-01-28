import re
import spacy
import string
import pandas as pd
import datetime as dt
sp = spacy.load('en_core_web_sm')


def main():
    rleaves = pd.read_csv('rleaves.csv', encoding='utf-8')
    rleaves[['day', 'hour']] = rleaves['time'].apply(change_time)
    rleaves['raw'] = pd.DataFrame(rleaves['raw'].apply(cleanup))
    rleaves = rleaves[['raw', 'day', 'hour']]
    rleaves.to_csv('rleaves_clean.csv', index=False)

def cleanup(text):
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

def change_time(utc):
    day = dt.datetime.fromtimestamp(utc).strftime('%A')
    hour = dt.datetime.fromtimestamp(utc).strftime('%I %p')
    return pd.Series([day, hour])

if __name__ == "__main__":
    main()