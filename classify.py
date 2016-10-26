"""
Train a Naive Bayes and a Bayesian logistic regression model, then test and use to classify unlabeled reviews
as to whether they judge the service.
"""

import pandas as pd
import numpy as np
import nltk
from nltk.classify import NaiveBayesClassifier
import sklearn
import sys

def main():
    data_directory = '../Data/'
    infile = data_directory + 'reviews_clean.csv'
    training_file = data_directory + 'reviews_training.csv'
    outfile = data_directory + 'reviews_classified.csv'
    outfile_small = data_directory + 'reviews_classified_small.csv'
    classification_file = '../Output/classification.txt'

    df_small = pd.read_csv(infile, nrows = 100)
    cols = [c for c in df_small.columns if 'hours_' not in c]
    cols = [c for c in cols if 'attributes_' not in c]
    df = pd.read_csv(infile, usecols = cols)
    training_df = pd.read_csv(training_file)
    training_df = training_df.drop('text', axis = 1)
    df = df.merge(training_df, on = ['review_id'], how = 'left')

    df = classify(df, classification_file)
    df.to_csv(outfile, index = False)
    df[0:200].to_csv(outfile_small, index = False)

def classify(df, classification_file):
    # df['yelping_since_year'] = df['yelping_since'].str[0: 4].astype(float)
    
    independent_vars = ['word_count', 'service_present', 'food_present', 'good_present', 'bad_present', 'money_present', 'size_present',
    'food_poisoning_present', 'speed_present', 'cleanliness_present']

    # Standardize independent variables (subtract mean and divide by standard deviation)
    df[independent_vars] = standardize(df[independent_vars])

    coded_rows = 101
    training_size = int(round(coded_rows * 0.7))
    test_size = int(round(coded_rows * 0.3))

    orig_stdout = sys.stdout
    f = file(classification_file, 'w')
    sys.stdout = f
    print 'Classification Results for dataframe with ' + str(coded_rows) + ' coded training rows\n'

    dependent_vars = ['service', 'food', 'money']
    for dependent_var in dependent_vars:
        df = naive_bayes(df, independent_vars, dependent_var, training_size, test_size)
        df = logistic_regression(df, independent_vars, dependent_var, training_size, test_size)

    sys.stdout = orig_stdout
    f.close()

    return df

def naive_bayes(df, independent_vars, dependent_var, training_size, test_size):
    def classify_review(row_number):
        """ Predict uncoded reviews' depdendent variable """
        return classifier.classify(features_dict[row_number])

    def label_probability(row_number):
        """ Predict probability of the outcome """
        outcome = 1
        return classifier.prob_classify(features_dict[row_number]).prob(outcome)

    # Create features dictionary, with each review ID (row number) as key, and features dict as value
    features_dict = df[independent_vars].to_dict('review_id')
    dependent_var_dict = df[[dependent_var]].to_dict('review_id')

    # Create (features_dict, label) tuples, e.g. ({'word_count': 10, 'mexican_count': 2}, 1)
    featuresets = [(features_dict[row_number], dependent_var_dict[row_number][dependent_var]) for row_number in df.index]
    train_set, test_set = featuresets[:training_size], featuresets[training_size: training_size + test_size]

    classifier = nltk.NaiveBayesClassifier.train(train_set)

    print 'Naive Bayes % test examples classified correctly for ' + dependent_var + '\n'
    print nltk.classify.accuracy(classifier, test_set) * 100
    print '\n'

    predicted_var = dependent_var + '_pred_nbayes'
    probability_var = dependent_var + '_prob_nbayes'
    df[predicted_var] = df.index.map(classify_review)
    df[probability_var] = df.index.map(label_probability)
    
    correctly_classified = dependent_var + '_correct_nbayes'
    df[correctly_classified] = (df[predicted_var] == df[dependent_var]).astype(float)
    df.loc[df[dependent_var].isnull(), correctly_classified] = np.nan
    df.loc[df[predicted_var].isnull(), correctly_classified] = np.nan

    return df

def logistic_regression(df, independent_vars, dependent_var, training_size, test_size):
    training_set = df[:training_size]
    training_set = training_set.dropna(subset = independent_vars + [dependent_var])

    # Train a logistic regression model with an L2 penalty
    # Alternative: use Bayesian logistic regression with a Bayesian prior distribution on the parameters
    logreg = sklearn.linear_model.LogisticRegression(penalty = 'l2', solver = 'sag')
    model = logreg.fit(training_set[independent_vars], training_set[dependent_var])

    print 'Odds ratio (exponentiated) coefficients via logistic regression:'
    print pd.DataFrame(zip(independent_vars, np.transpose(model.coef_)))
    print '\n'

    # Predicted value (binary)
    predicted_var = dependent_var + '_pred_logit'
    valid_rows = df[independent_vars].notnull().all(axis=1)
    df.loc[valid_rows, predicted_var] = model.predict(df[valid_rows][independent_vars])

    # Predicted probability that dependent variable is positive
    df.loc[valid_rows, dependent_var + '_pred_prob_logit'] = model.predict_proba(df[valid_rows][independent_vars])[:,1]

    correctly_classified = dependent_var + '_correct_logit'
    df[correctly_classified] = (df[predicted_var] == df[dependent_var]).astype(float)
    df.loc[(df[dependent_var].isnull()) | (df[predicted_var].isnull()), correctly_classified] = np.nan

    test_set = df[training_size: training_size + test_size]
    print 'Logistic regression % test examples classified correctly for ' + dependent_var
    print test_set[correctly_classified].mean() * 100
    print '\n'

    return df

def standardize(df):
    for col in df.columns:
        # subtract the mean
        col_mean = df[col].mean()
        df[col] = df[col] - col_mean

        # divide by standard deviation
        df[col] = df[col] / df[col].std()
    return df

if __name__ == '__main__':
    main()