import pandas as pd
import textmining
import clean_nltk as cn

def main():
    data_directory = "../Data/"
    infile = data_directory + "reviews_clean.csv"
    outfile = data_directory + "term_count.csv"
    df = pd.read_csv(infile)

    terms_matrix = textmining.TermDocumentMatrix()
    categories = ['Mexican', 'Italian', 'American']
    for category in categories:
        subset = df[df['category'] == category]
        raw_text = ' '.join(subset['text'].astype(str).tolist()).lower()
        text = cn.parse_text(raw_text)
        terms_matrix.add_doc(text)

    terms_df = make_df(terms_matrix)
    terms_df.to_csv(outfile, index = True)

def make_df(matrix):
    term_data = pd.DataFrame(matrix.rows())
    term_data.columns = term_data.iloc[0]
    term_data = term_data[1:]
    term_data.index = categories
    term_data.index.name = 'category'
    term_data = term_data.T
    term_data.index.name = 'term'
    return term_data

if __name__ == "__main__":
    main()