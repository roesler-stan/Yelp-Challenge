"""
Train a Naive Bayes and a Bayesian logistic regression model, then test and use to classify unlabeled reviews
as to whether they judge the service.
"""

import pandas as pd
import numpy as np
import nltk
from nltk.classify import NaiveBayesClassifier
import sys
import re
from nltk.corpus import stopwords
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import lda

def main():
    data_directory = '../Data/'
    infile = data_directory + 'reviews_clean.csv'
    labeled_file = data_directory + 'reviews_labeled.csv'
    outfile = data_directory + 'reviews_classified.csv'
    outfile_small = data_directory + 'reviews_classified_small.csv'
    classification_file = '../Output/classification.txt'
    logistic_file = '../Output/logistic_reg_coefs.txt'
    bow_file = '../Output/classification_bow.txt'
    bow_logistic_file = '../Output/logistic_reg_coefs_bow.txt'

    TEST_PROPORTION = 0.3
    TOPICS = ['service', 'food', 'money']

    df_small = pd.read_csv(infile, nrows = 100)
    cols = [c for c in df_small.columns if 'hours_' not in c]
    cols = [c for c in cols if 'attributes_' not in c]
    df = pd.read_csv(infile, usecols = cols)
    labeled_df = pd.read_csv(labeled_file)
    labeled_df = labeled_df.dropna(subset = TOPICS)
    
    df = bag_of_words(labeled_df, df, bow_file, bow_logistic_file, TOPICS, TEST_PROPORTION)

    labeled_df = labeled_df.drop('text', axis = 1)
    df = df.merge(labeled_df, on = ['review_id'], how = 'left')

    df = classify(df, classification_file, logistic_file, TOPICS, TEST_PROPORTION)
    df.to_csv(outfile, index = False)
    df[0:200].to_csv(outfile_small, index = False)

def bag_of_words(labeled_df, df, outfile, logistic_file, TOPICS, TEST_PROPORTION):
    labeled_df['words'] = labeled_df['text'].map(review_to_words)
    TRAINING_ROWS = int(len(labeled_df) * (1 - TEST_PROPORTION))
    TEST_ROWS = int(len(labeled_df) * TEST_PROPORTION)
    training_df = labeled_df[:TRAINING_ROWS]
    test_df = labeled_df[TRAINING_ROWS: TRAINING_ROWS + TEST_ROWS]

    training_words = training_df['words'].tolist()
    vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000) 
    training_data_features = vectorizer.fit_transform(training_words)
    training_data_features = training_data_features.toarray()
    training_data_features_std = standardize_array(training_data_features)

    test_words = test_df['words'].tolist()
    test_data_features = vectorizer.transform(test_words)
    test_data_features = test_data_features.toarray()
    test_data_features_std = standardize_array(test_data_features)

    df['words'] = df['text'].map(review_to_words)
    words = df['words'].tolist()
    features = vectorizer.transform(words)
    features = features.toarray()
    features_std = standardize_array(features)

    orig_stdout = sys.stdout
    f = file(outfile, 'w')
    sys.stdout = f
    print('Bag of words test set accuracies\n')

    for topic in TOPICS:
        # Initialize a Random Forest classifier with 100 trees
        forest = RandomForestClassifier(n_estimators = 100)
        # Fit the forest to the labeled set, using the bag of words as features and the sentiment labels as the response variable
        # This may take a few minutes to run
        forest = forest.fit( training_data_features_std, training_df[topic] )
        result = forest.predict(test_data_features_std)
        print(topic + '\n')
        print('Random forest: ' + str(round((np.mean(result == test_df[topic]) * 100), 2)) + '\n')
        df[topic + '_pred_bow_randforest'] = forest.predict(features_std)

        clf = sklearn.svm.SVC()
        clf.fit(training_data_features_std, training_df[topic])
        predicted = clf.predict(test_data_features_std)
        print('SVM: ' + str(round(((predicted == test_df[topic]).mean() * 100), 2)))
        df[topic + '_pred_bow_svm'] = clf.predict(features_std)

        logreg = sklearn.linear_model.LogisticRegression(penalty = 'l2', solver = 'sag')
        model = logreg.fit(training_data_features_std, training_df[topic])
        message = topic + ':\n' + str(pd.DataFrame(zip(vectorizer.get_feature_names(), np.transpose(model.coef_)))) + '\n\n'
        with open(logistic_file, 'a') as outf:
            outf.write(message)

        predicted = model.predict(test_data_features_std)
        print('Logistic regression: ' + str(round((predicted == test_df[topic]).mean() * 100, 2)))
        df[topic + '_pred_logit'] = model.predict(features_std)
        df[topic + '_pred_prob_logit'] = model.predict_proba(features_std)[:,1]

    df = df.drop('words', axis=1)

    sys.stdout = orig_stdout
    f.close()
    return df

def review_to_words(review):
    """ Function to convert a raw review to a string of comparable words """
    letters_only = re.sub("[^a-zA-Z]", " ", str(review))
    # Convert to lower case, split into individual words
    words = letters_only.lower().split()
    # Remove stop words
    # In Python, searching a set is much faster than searching a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    meaningful_words = [w for w in words if not w in stops]
    clean_review = " ".join(meaningful_words)
    return(clean_review)

def classify(df, classification_file, logistic_file, TOPICS, TEST_PROPORTION):
    CODED_ROWS = sum(~df[TOPICS].isnull().any(1))
    independent_vars = ['word_count', 'service_present', 'food_present', 'money_present']
    independent_vars_std = [v + '_std' for v in independent_vars]

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
    print('Test set accuracy for dataframe with ' + str(CODED_ROWS) + ' coded labeled rows and ' + str(TEST_PROPORTION * 100) + '% test set.')

    for topic in TOPICS:
        current_features = ['word_count_std', topic + '_present_std']
        # current_features = independent_vars_std

        training_set = df[:training_size]
        test_set = df[training_size: training_size + test_size]

        X_train = training_set[current_features]
        y_train = training_set[topic]
        X_test = test_set[current_features]
        y_test = test_set[topic]
        X_full = df[current_features]

        print('\n\n' + topic + ':\n')
        df = svm(df, X_train, y_train, X_test, y_test, X_full, topic)
        df = logistic_regression(df, X_train, y_train, X_test, y_test, X_full, current_features, topic, logistic_file)
        df = naive_bayes(df, current_features, topic, labeled_size, test_size)

    sys.stdout = orig_stdout
    f.close()

    return df

def svm(df, X_train, y_train, X_test, y_test, X_full, topic):
    clf = sklearn.svm.SVC()
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)

    print('SVM: ' + str(round(((predicted == y_test).mean() * 100), 2)))

    predicted_var = topic + '_pred_svm'
    df[predicted_var] = clf.predict(X_full)
    return df

def naive_bayes(df, current_features, topic, labeled_size, test_size):
    def classify_review(row_number):
        """ Predict uncoded reviews' depdendent variable """
        return classifier.classify(features_dict[row_number])

    def label_probability(row_number):
        """ Predict probability of the outcome """
        return classifier.prob_classify(features_dict[row_number]).prob(1)

    # Create features dictionary, with each review ID (row number) as key, and features dict as value
    features_dict = df[current_features].to_dict('review_id')
    topic_dict = df[[topic]].to_dict('review_id')

    # Create (features_dict, label) tuples, e.g. ({'word_count': 10, 'mexican_count': 2}, 1)
    featuresets = [(features_dict[row_number], topic_dict[row_number][topic]) for row_number in df.index]
    train_set, test_set = featuresets[:labeled_size], featuresets[labeled_size: labeled_size + test_size]

    classifier = nltk.NaiveBayesClassifier.train(train_set)

    print('Naive Bayes: ' + str(round(nltk.classify.accuracy(classifier, test_set) * 100, 2)))

    predicted_var = topic + '_pred_nbayes'
    probability_var = topic + '_prob_nbayes'
    df[predicted_var] = df.index.map(classify_review)
    df[probability_var] = df.index.map(label_probability)

    return df

def logistic_regression(df, X_train, y_train, X_test, y_test, X_full, current_features, topic, logistic_file):
    # Train a logistic regression model with an L2 penalty
    # Alternative: use Bayesian logistic regression with a Bayesian prior distribution on the parameters
    logreg = sklearn.linear_model.LogisticRegression(penalty = 'l2', solver = 'sag')
    model = logreg.fit(X_train, y_train)

    message = topic + ':\n' + str(pd.DataFrame(zip(current_features, np.transpose(model.coef_)))) + '\n\n'
    with open(logistic_file, 'a') as outf:
        outf.write(message)

    # Predicted value (binary)
    predicted_var = topic + '_pred_logit'
    predicted = model.predict(X_test)
    print('Logistic regression: ' + str(round((predicted == y_test).mean() * 100, 2)))

    df[predicted_var] = model.predict(X_full)

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

def standardize_array(arr):
    means = np.mean(arr, 0)
    stds = np.std(arr, 0)
    new_arr = (arr - means) / stds
    new_arr = np.nan_to_num(new_arr)
    return new_arr

def lda_topics(training_data_features):
    model = lda.LDA(n_topics=20, n_iter=1500, random_state=1)
    model.fit(training_data_features)
    topic_word = model.topic_word_  # model.components_ also works
    n_top_words = 8
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vectorizer.get_feature_names())[np.argsort(topic_dist)][:-(n_top_words+1):-1]
        print('Topic {}: {}'.format(i, ' '.join(topic_words)))
    # Topic with highest likelihood
    predicted = model.transform(test_data_features).max(axis=1)

if __name__ == '__main__':
    main()