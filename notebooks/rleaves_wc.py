import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from wordcloud import WordCloud, STOPWORDS


def main():
    df = pd.read_csv('rleaves_clean.csv', encoding='utf-8')
    wordcloud_text = ' '.join(df['raw'].tolist())
    mask = np.array(Image.open('marijuana.png'))
    wordcloud_user = WordCloud(width=300, height=200, background_color='rgba(255, 255, 255, 0)', mode='RGBA', colormap='tab10', collocations=False, mask=mask, include_numbers=True)
    wordcloud_user.generate(wordcloud_text)
    #wordcloud_user.to_file("wordcloud_user_leaves.png") # saves the image
    plot_cloud(wordcloud_user)

def plot_cloud(wordcloud):
    plt.figure(figsize=(40, 30))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    
if __name__ == "__main__":
    main()