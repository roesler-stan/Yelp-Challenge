"""
TO DO: analyze for money, too
Calculate the stars rating * probability that was each category?
"""
import pandas as pd
import numpy as np
import csv
import sys
import matplotlib.pyplot as plt

def main():
    infile = '../Data/reviews_classified.csv'
    outfile = '../Data/reviews_classified_clean.csv'
    df = pd.read_csv(infile)

    df = describe(df, describe_file)
    df.to_csv(outfile, index = False, quoting = csv.QUOTE_MINIMAL, quotechar = '"')

def find_outliers(df):
    (df['stars_review_service_standardized'] > 2).mean()
    (df['stars_review_service_standardized'] < -2).mean()
    bad_service = df[df['stars_review_service_standardized'] < -2]
    
    pd.options.display.max_colwidth = 500
    bad_service['text'][:3]

    bad_food = df[df['stars_review_food_standardized'] < -2]
    bad_food['text'][:2]

    df[df['food_count'] > 3]['text'][:3]

def describe(df, outfile):
    orig_stdout = sys.stdout
    f = file(outfile, 'w')
    sys.stdout = f

    df['stars_review_service_mean'].mean()
    df['stars_review_non_service_mean'].mean()
    df['stars_review_food_mean'].mean()
    df['stars_review_non_food_mean'].mean()

    (df['stars_review_service_mean'] / df['stars_review_non_service_mean']).mean()
    (df['stars_review_food_mean'] / df['stars_review_non_food_mean']).mean()
    print "If they talk about food, it's really good for stars.\n"
    print "If they talk about service, it's bad for stars.\n"

    # Replace 0 so there's no division by zero - do to all to be consistent
    cols = ['polarity_service_mean', 'polarity_non_service_mean', 'polarity_food_mean', 'polarity_non_food_mean']
    for col in cols:
        df.loc[df[col] == 0, col] = 0.0001

    df['polarity_service_mean'].mean()
    df['polarity_non_service_mean'].mean()
    df['polarity_food_mean'].mean()
    df['polarity_non_food_mean'].mean()

    print "The difference is even more striking for review polarity, rather than stars."

    (df['polarity_service_mean'] / df['polarity_non_service_mean']).mean()
    (df['polarity_food_mean'] / df['polarity_non_food_mean']).mean()

    sys.stdout = orig_stdout
    f.close()

def calculate_groups(df):
    cols = []
    outcomes = ['stars_review', 'polarity']
    categories = ['service', 'food', 'money']
    for outcome in outcomes:
        for category in categories:
            positive = df.loc[df[category + '_pred_logit'] == 1].groupby('business_id')[outcome].agg([np.mean, np.std]).reset_index()
            negative = df.loc[df[category + '_pred_logit'] == 0].groupby('business_id')[outcome].agg([np.mean, np.std]).reset_index()

            pos_mean_col = '_'.join([outcome, category, 'mean'])
            neg_mean_col = '_'.join([outcome, 'non', category, 'mean'])
            cols += [pos_mean_col, neg_mean_col]
            positive = positive.rename(columns = {'mean': mean_col, 'std': '_'.join([outcome, category, 'std'])})
            negative = negative.rename(columns = {'mean': neg_mean_col, 'std': '_'.join([outcome, 'non', category, 'std'])})

            df = df.merge(positive, on = ['business_id'], how = 'left')
            df = df.merge(negative, on = ['business_id'], how = 'left')

    for col in cols:
        df[col.split('_mean')[0] + '_standardized'] = (df[col] - df[col].mean()) / df[col].std()

    return df

if __name__ == "__main__":
	main()