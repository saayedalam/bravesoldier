import textacy.ke
import pandas as pd
from textacy import *

def main():
    df = pd.read_csv('rleaves_clean.csv', encoding='utf-8')
    textrank = get_top_ranked(df['raw'])
    textrank.to_csv('rleaves_textrank.csv', index=False)
    
def get_top_ranked(data):
    en = textacy.load_spacy_lang('en_core_web_sm')
    text = ' '.join(data)
    doc = textacy.make_spacy_doc(text, lang=en)
    df = pd.DataFrame(textacy.ke.yake(doc, ngrams=2, window_size=4, topn=10)).rename(columns={0: "Most Important Keywords", 1: "YAKE! Score"})
    return df

if __name__ == "__main__":
    main()