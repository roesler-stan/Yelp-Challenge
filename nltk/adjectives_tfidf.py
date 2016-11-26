import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import clean_nltk as cn

def main():
    data_directory = "../Data/"
    infile = data_directory + "reviews_clean.csv"
    outfile = data_directory + "adjectives_tfidf.csv"
    df = pd.read_csv(infile)

    category_dict = {}
    categories = ['Mexican', 'Italian', 'American']
    for category in categories:
        subset = df[df['category'] == category]
        text = cn.select_text(subset['text'])
        entities = cn.parse_text(text, lemmatize = False)
        adjectives = [str(entity[0]) for entity in entities if entity[-1] == 'JJ']
        adjectives_text = ' '.join(adjectives)
        category_dict[category] = adjectives_text

    tfidf = TfidfVectorizer()
    tfs = tfidf.fit_transform(category_dict.values())
    tfidf_data = pd.DataFrame([ pd.SparseSeries(tfs[i].toarray().ravel()) for i in np.arange(tfs.shape[0]) ])
    columns = tfidf.get_feature_names()
    tfidf_data.columns = columns
    tfidf_data.index = category_dict.keys()

    tfidf_data = tfidf_data.stack().reset_index()
    tfidf_data = tfidf_data.rename(columns = {'level_0': 'category', 'level_1': 'term', 0: 'tfidf'})
    top_data = tfidf_data.sort(['category', 'tfidf'], ascending = False).groupby('category').head()
    top20_data = tfidf_data.sort(['category', 'tfidf'], ascending = False).groupby('category').head(20)
    top20_data.to_csv(outfile, index = False)

if __name__ == "__main__":
    main()