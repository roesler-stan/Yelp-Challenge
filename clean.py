import pandas as pd
import numpy as np
import code_reviews as cr
import csv

def main():
    outfile = '../Data/reviews_clean.csv'
    df = read_data()
    df = clean_data(df)
    # For some reason, it has to be index=True
    df.to_csv(outfile, index = True, quoting = csv.QUOTE_MINIMAL, quotechar = '"')

    # Hand-code this later and rename reviews_training.csv
    df[['review_id', 'text']][: 1000].to_csv('../Data/reviews_only_small.csv', index = False)
    df[['review_id', 'text']].to_csv('../Data/reviews_only.csv', index = False)

def read_data():
    filename_base = '../Data/input/yelp_academic_dataset'
    file_business = filename_base + '_business.csv'
    file_review = filename_base + '_review.csv'
    file_user = filename_base + '_user.csv'

    df_business = pd.read_csv(file_business)
    df_review = pd.read_csv(file_review)
    df_user = pd.read_csv(file_user)

    # Merge the datasets, keeping them on the review level
    # Each review has one user and is for one business, so we use left outer joins
    df = df_review.merge(df_user, on = ['user_id'], how = 'left', suffixes = ['_review', '_user'])
    df = df.merge(df_business, on = ['business_id'], how = 'left', suffixes = ['_review', '_business'])
    df = df.rename(columns = lambda x: x.replace('.', '_'))

    return df

def clean_data(df):
    # Only include restaurants
    df = df[df['categories'].str.contains('restaurant', case = False, na = False)]

    df['date'] = pd.to_datetime(df['date'], yearfirst = True, errors = 'coerce')
    # df['date'] = pd.to_datetime(df['date'], errors = 'coerce', format = '%Y-%m-%d')
    df['year'] = df['date'].dt.year

    df = clean_attributes(df)

    # Decode non-Ascii text
    df['text'] = df['text'].str.decode('unicode_escape', errors = 'ignore').str.encode('ascii', errors = 'ignore')
    df = cr.categories(df)
    df = cr.themes(df)
    df = cr.languages(df)

    # Shuffle the data, using the same seed to be able to keep track of the data
    np.random.seed(72)
    df = df.reindex(np.random.permutation(df.index))

    return df

def clean_attributes(df):
    # Convert all attributes booleans to floats: also considers 1 and 0 to be boolean
    for col in [c for c in df.columns if 'attributes' in c]:
        col_values = df[col][:100].unique().tolist()
        if True in col_values or False in col_values:
            df[col] = df[col].astype(bool).astype(float)
    return df

if __name__ == "__main__":
    main()