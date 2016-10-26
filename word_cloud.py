""" Draw word cloud using data from Yelp.com reviews of Mexican restaurants """

import pandas as pd
import numpy as np
from PIL import Image
import wordcloud as wc
import clean_nltk as cn

def main():
    infile = '../Data/reviews_only.csv'
    mask_file = '../Output/speech_bubble.png'
    outfile_adj = '../Output/yelp_adj.png'

    # Only read the first 100K rows, since anyway it will only pick the 2,000 most common adjectives
    df = pd.read_csv(infile, nrows = 100000)

    text = cn.select_text(df['text'])
    entities = cn.parse_text(text, lemmatize = False)
    adjectives = [str(entity[0]) for entity in entities if entity[-1] == 'JJ']
    adjectives = [ 'delicious' if adj == 'deliciou' else adj for adj in adjectives ]
    adjectives_text = ' '.join(adjectives)

    max_words = 2000
    masked_cloud(adjectives_text, outfile_adj, max_words, mask_file)

def masked_cloud(text, outfile, max_words, mask_file):
    """ Make a word cloud in the shape of the mask file's black parts """
    mask_shape = np.array(Image.open(mask_file))
    word_cloud = wc.WordCloud(max_words = max_words, background_color = "white", mask = mask_shape)
    # stopwords = wc.STOPWORDS.add("said")    
    word_cloud.generate(text)
    word_cloud.to_file(outfile)

if __name__ == '__main__':
    main()