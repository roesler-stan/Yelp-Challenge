"""
Train a Naive Bayes and a Bayesian logistic regression model, then test and use to classify unlabeled reviews
as to whether they judge the service.
"""

import pandas as pd
import numpy as np
import sys
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


def main():
    data_directory = '../Data/'
    infile = data_directory + 'reviews_clean.csv'
    labeled_file = data_directory + 'reviews_labeled.csv'
    outfile = data_directory + 'reviews_classified.csv'
    outfile_small = data_directory + 'reviews_classified_small.csv'
    classification_file = '../Output/classification.txt'
    logistic_file = '../Output/logistic_reg_coefs.txt'

    TEST_PROPORTION = 0.3
    TOPICS = ['food', 'service', 'money']

    df_small = pd.read_csv(infile, nrows = 100)
    cols = [c for c in df_small.columns if 'hours_' not in c]
    cols = [c for c in cols if 'attributes_' not in c]
    df = pd.read_csv(infile, usecols = cols)
    labeled_reviews = pd.read_csv(labeled_file)
    labeled_reviews = labeled_reviews.dropna(subset = TOPICS)
    labeled_reviews = labeled_reviews.drop('text', axis = 1)
    df = df.merge(labeled_reviews, on = ['review_id'], how = 'left')

    df = classify(df, classification_file, logistic_file, TOPICS, TEST_PROPORTION)
    df.to_csv(outfile, index = False)
    df[0:200].to_csv(outfile_small, index = False)


def classify(df, classification_file, logistic_file, TOPICS, TEST_PROPORTION):
    independent_vars = ['word_count', 'service_present', 'food_present', 'money_present']
    df = df.dropna(subset = independent_vars)
    df = df.reset_index()

    labeled_df = df[~df[TOPICS].isnull().any(1)].copy()
    CODED_ROWS = len(labeled_df)
    training_size = int(round(CODED_ROWS * (1 - TEST_PROPORTION)))
    test_size = int(round(CODED_ROWS * TEST_PROPORTION))
    training_set = labeled_df[:training_size]
    test_set = labeled_df[training_size: training_size + test_size]

    with open(logistic_file, 'w') as outf:
        outf.write("Measures' Odds ratio (exponentiated) coefficients via logistic regression\n\n")

    orig_stdout = sys.stdout
    f = file(classification_file, 'w')
    sys.stdout = f
    print('Test set accuracy for dataframe with ' + str(CODED_ROWS) + ' coded labeled rows and a ' + str(TEST_PROPORTION * 100) + '% test set.')

    for topic in TOPICS:
        current_features = [topic + '_present']
        # current_features = ['word_count', topic + '_present']
        X_train = training_set[current_features].copy()
        y_train = training_set[topic].copy()
        X_test = test_set[current_features].copy()
        y_test = test_set[topic].copy()
        X_full = df[current_features].copy()

        X_train = standardize(X_train)
        X_test = standardize(X_test)
        X_full = standardize(X_full)

        print('\n\n' + topic + ':\n')
        df = svm(df, X_train, y_train, X_test, y_test, X_full, topic)
        df = naive_bayes(df, X_train, y_train, X_test, y_test, X_full, topic)
        df = logistic_regression(df, X_train, y_train, X_test, y_test, X_full, current_features, topic, logistic_file)

    sys.stdout = orig_stdout
    f.close()

    return df


def svm(df, X_train, y_train, X_test, y_test, X_full, topic):
    clf = sklearn.svm.SVC()
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    print('SVM: ' + str(round(((predicted == y_test).mean() * 100), 4)))
    df[topic + '_pred_svm'] = clf.predict(X_full)
    return df


def naive_bayes(df, X_train, y_train, X_test, y_test, X_full, topic):
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    print('Naive Bayes: ' + str(round(((predicted == y_test).mean() * 100), 4)))
    df[topic + '_pred_nbayes'] = clf.predict(X_full)
    return df


def logistic_regression(df, X_train, y_train, X_test, y_test, X_full, current_features, topic, logistic_file):
    """ Train a logistic regression model with an L2 penalty """
    logreg = sklearn.linear_model.LogisticRegression(penalty = 'l2', solver = 'sag')
    model = logreg.fit(X_train, y_train)

    message = topic + ':\n' + str(pd.DataFrame(zip(current_features, np.transpose(model.coef_)))) + '\n\n'
    with open(logistic_file, 'a') as outf:
        outf.write(message)

    # Predicted value (binary)
    predicted = model.predict(X_test)
    print('Logistic regression: ' + str(round((predicted == y_test).mean() * 100, 4)))
    df[topic + '_pred_logit'] = model.predict(X_full)

    # Predicted probability that dependent variable is positive
    df[topic + '_pred_prob_logit'] = model.predict_proba(X_full)[:,1]

    return df


def standardize(df):
    """ Standardize dataframe columns """
    for col in df.columns:
        # subtract the mean
        col_mean = df[col].mean()
        df[col] = df[col] - col_mean
        # divide by standard deviation
        df[col] = df[col] / df[col].std()
    return df

if __name__ == '__main__':
    main()