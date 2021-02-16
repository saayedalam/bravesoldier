import re
import pandas as pd
from string import punctuation
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def main():
    rleaves = pd.read_csv('rleaves.csv', encoding='utf-8')
    rleaves = rleaves['raw']
    rleaves = rleaves.apply(cleanup)
    
    emotions = rleaves.apply(get_emotion)
    emotions = emotions.str.replace('<pad> |</s>', '')
    
    rleaves.name = 'post'
    emotions.name = 'emotion'
    rleaves_emotion = pd.concat([rleaves, emotions], axis=1)
    rleaves_emotion.to_csv('rleaves_emotion.csv', index=False)

def cleanup(text):
    text = text.lower() # lowers the corpus
    text = re.sub('http\S+', ' ', str(text)) # removes any url
    text = re.sub('n\'t\s', ' not ', str(text)) # change apostrophe
    text = re.sub('-(?<!\d)', ' ', str(text)) # removing hyphens from numbers
    my_punctuation = punctuation.replace(".", "") # removes all punctuation except period
    text = text.translate(str.maketrans('', '', my_punctuation))
    text = re.sub('’|“|”|\.{2,}', '', str(text))
    text = re.sub('x200b', ' ', str(text)) # removing zero-width space characters
    return ' '.join([token for token in text.split()]) # removes trailing whitespaces

def get_emotion(text):
    tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-emotion")
    model = AutoModelForSeq2SeqLM.from_pretrained("mrm8488/t5-base-finetuned-emotion")
    input_ids = tokenizer.encode(text, return_tensors='pt')
    output = model.generate(input_ids=input_ids)
    dec = [tokenizer.decode(ids) for ids in output]
    label = dec[0]
    return label

if __name__ == "__main__":
    main()