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
    logistic_file = '../Output/logistic_reg_coefs.txt'

    df_small = pd.read_csv(infile, nrows = 100)
    cols = [c for c in df_small.columns if 'hours_' not in c]
    cols = [c for c in cols if 'attributes_' not in c]
    df = pd.read_csv(infile, usecols = cols)
    training_df = pd.read_csv(training_file)
    training_df = training_df.drop('text', axis = 1)
    df = df.merge(training_df, on = ['review_id'], how = 'left')

    CODED_ROWS = len(training_df['food'].notnull())
    TEST_PROPORTION = 0.3
    df = classify(df, classification_file, CODED_ROWS, TEST_PROPORTION, logistic_file)
    df.to_csv(outfile, index = False)
    df[0:200].to_csv(outfile_small, index = False)

def classify(df, classification_file, CODED_ROWS, TEST_PROPORTION, logistic_file):
    independent_vars = ['word_count', 'service_present', 'food_present', 'money_present']
    independent_vars_std = [v + '_std' for v in independent_vars]
    dependent_vars = ['service', 'food', 'money']

    df = df.dropna(subset = independent_vars)
    df = df.reset_index()
    # Standardize independent variables (subtract mean and divide by standard deviation)
    df[independent_vars_std] = standardize(df[independent_vars])

    training_size = int(round(CODED_ROWS * (1 - TEST_PROPORTION)))
    test_size = int(round(CODED_ROWS * TEST_PROPORTION))

    message = 'Odds ratio (exponentiated) coefficients via logistic regression\n\n'
    with open(logistic_file, 'w') as outf:
        outf.write(message)

    orig_stdout = sys.stdout
    f = file(classification_file, 'w')
    sys.stdout = f
    print 'Test set accuracy for dataframe with ' + str(CODED_ROWS) + ' coded training rows and ' + str(TEST_PROPORTION * 100) + '% test set.'

    for dependent_var in dependent_vars:
        current_features = ['word_count_std', dependent_var + '_present_std']
        # current_features = independent_vars_std

        training_set = df[:training_size]
        test_set = df[training_size: training_size + test_size]

        training_set  = training_set.dropna(subset = dependent_vars)
        test_set  = test_set.dropna(subset = dependent_vars)

        X_train = training_set[current_features]
        y_train = training_set[dependent_var]
        X_test = test_set[current_features]
        y_test = test_set[dependent_var]
        X_full = df[current_features]

        print '\n\n' + dependent_var + ':\n'
        df = svm(df, X_train, y_train, X_test, y_test, X_full, dependent_var)
        df = logistic_regression(df, X_train, y_train, X_test, y_test, X_full, current_features, dependent_var, logistic_file)
        df = naive_bayes(df, current_features, dependent_var, training_size, test_size)

    sys.stdout = orig_stdout
    f.close()

    return df

def svm(df, X_train, y_train, X_test, y_test, X_full, dependent_var):
    clf = sklearn.svm.SVC()
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)

    print 'SVM: ' + str(round(((predicted == y_test).mean() * 100), 2))

    predicted_var = dependent_var + '_pred_svm'
    df[predicted_var] = clf.predict(X_full)
    return df

def naive_bayes(df, current_features, dependent_var, training_size, test_size):
    def classify_review(row_number):
        """ Predict uncoded reviews' depdendent variable """
        return classifier.classify(features_dict[row_number])

    def label_probability(row_number):
        """ Predict probability of the outcome """
        return classifier.prob_classify(features_dict[row_number]).prob(1)

    # Create features dictionary, with each review ID (row number) as key, and features dict as value
    features_dict = df[current_features].to_dict('review_id')
    dependent_var_dict = df[[dependent_var]].to_dict('review_id')

    # Create (features_dict, label) tuples, e.g. ({'word_count': 10, 'mexican_count': 2}, 1)
    featuresets = [(features_dict[row_number], dependent_var_dict[row_number][dependent_var]) for row_number in df.index]
    train_set, test_set = featuresets[:training_size], featuresets[training_size: training_size + test_size]

    classifier = nltk.NaiveBayesClassifier.train(train_set)

    print 'Naive Bayes: ' + str(round(nltk.classify.accuracy(classifier, test_set) * 100, 2))

    predicted_var = dependent_var + '_pred_nbayes'
    probability_var = dependent_var + '_prob_nbayes'
    df[predicted_var] = df.index.map(classify_review)
    df[probability_var] = df.index.map(label_probability)

    return df

def logistic_regression(df, X_train, y_train, X_test, y_test, X_full, current_features, dependent_var, logistic_file):
    # Train a logistic regression model with an L2 penalty
    # Alternative: use Bayesian logistic regression with a Bayesian prior distribution on the parameters
    logreg = sklearn.linear_model.LogisticRegression(penalty = 'l2', solver = 'sag')
    model = logreg.fit(X_train, y_train)

    message = dependent_var + ':\n' + str(pd.DataFrame(zip(current_features, np.transpose(model.coef_)))) + '\n\n'
    with open(logistic_file, 'a') as outf:
        outf.write(message)

    # Predicted value (binary)
    predicted_var = dependent_var + '_pred_logit'
    predicted = model.predict(X_test)
    print 'Logistic regression: ' + str(round((predicted == y_test).mean() * 100, 2))

    df[predicted_var] = model.predict(X_full)

    # Predicted probability that dependent variable is positive
    df[dependent_var + '_pred_prob_logit'] = model.predict_proba(X_full)[:,1]

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