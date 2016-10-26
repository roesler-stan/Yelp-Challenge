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
    (df['stars_service_standardized'] > 2).mean()
    (df['stars_service_standardized'] < -2).mean()
    bad_service = df[df['stars_service_standardized'] < -2]
    
    pd.options.display.max_colwidth = 500
    bad_service['text'][:3]

    bad_food = df[df['stars_food_standardized'] < -2]
    bad_food['text'][:2]

    df[df['food_count'] > 3]['text'][:3]

def describe(df, outfile):
    orig_stdout = sys.stdout
    f = file(outfile, 'w')
    sys.stdout = f

    df['stars_service_mean'].mean()
    df['stars_non_service_mean'].mean()
    df['stars_food_mean'].mean()
    df['stars_non_food_mean'].mean()

    (df['stars_service_mean'] / df['stars_non_service_mean']).mean()
    (df['stars_food_mean'] / df['stars_non_food_mean']).mean()
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
    service_stars = df.loc[df['service_pred_logit'] == 1].groupby('business_id')['stars_review'].agg([np.mean, np.std]).reset_index()
    non_service_stars = df.loc[df['service_pred_logit'] == 0].groupby('business_id')['stars_review'].agg([np.mean, np.std]).reset_index()
    food_stars = df.loc[df['food_pred_logit'] == 1].groupby('business_id')['stars_review'].agg([np.mean, np.std]).reset_index()
    non_food_stars = df.loc[df['food_pred_logit'] == 0].groupby('business_id')['stars_review'].agg([np.mean, np.std]).reset_index()

    service_stars = service_stars.rename(columns = {'mean': 'stars_service_mean', 'std': 'stars_service_std'})
    non_service_stars = non_service_stars.rename(columns = {'mean': 'stars_non_service_mean', 'std': 'stars_non_service_std'})
    food_stars = food_stars.rename(columns = {'mean': 'stars_food_mean', 'std': 'stars_food_std'})
    non_food_stars = non_food_stars.rename(columns = {'mean': 'stars_non_food_mean', 'std': 'stars_non_food_std'})

    service_polarity = df.loc[df['service_pred_logit'] == 1].groupby('business_id')['polarity'].agg([np.mean, np.std]).reset_index()
    non_service_polarity = df.loc[df['service_pred_logit'] == 0].groupby('business_id')['polarity'].agg([np.mean, np.std]).reset_index()
    food_polarity = df.loc[df['food_pred_logit'] == 1].groupby('business_id')['polarity'].agg([np.mean, np.std]).reset_index()
    non_food_polarity = df.loc[df['food_pred_logit'] == 0].groupby('business_id')['polarity'].agg([np.mean, np.std]).reset_index()

    service_polarity = service_polarity.rename(columns = {'mean': 'polarity_service_mean', 'std': 'polarity_service_std'})
    non_service_polarity = non_service_polarity.rename(columns = {'mean': 'polarity_non_service_mean', 'std': 'polarity_non_service_std'})
    food_polarity = food_polarity.rename(columns = {'mean': 'polarity_food_mean', 'std': 'polarity_food_std'})
    non_food_polarity = non_food_polarity.rename(columns = {'mean': 'polarity_non_food_mean', 'std': 'polarity_non_food_std'})

    df = df.merge(service_stars, on = ['business_id'], how = 'left')
    df = df.merge(non_service_stars, on = ['business_id'], how = 'left')
    df = df.merge(food_stars, on = ['business_id'], how = 'left')
    df = df.merge(non_food_stars, on = ['business_id'], how = 'left')
    df = df.merge(service_polarity, on = ['business_id'], how = 'left')
    df = df.merge(non_service_polarity, on = ['business_id'], how = 'left')
    df = df.merge(food_polarity, on = ['business_id'], how = 'left')
    df = df.merge(non_food_polarity, on = ['business_id'], how = 'left')

    cols = ['stars_service', 'stars_non_service', 'stars_food', 'stars_non_food',
    'polarity_service', 'polarity_non_service', 'polarity_food', 'polarity_non_food']
    for col in cols:
        df[col + '_standardized'] = (df[col + '_mean'] - df[col + '_mean'].mean()) / df[col + '_mean'].std()

    return df

if __name__ == "__main__":
	main()