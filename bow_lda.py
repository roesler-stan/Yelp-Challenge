"""
Bag-of-words assumption, Latent Dirichlet Analysis to classify reviews using their words
This usually runs out of RAM and gets killed.
"""

import pandas as pd
import numpy as np
import re
import sys
from nltk.corpus import stopwords
import lda

def main():
    data_directory = '../Data/'
    infile = data_directory + 'reviews_clean.csv'
    labeled_file = data_directory + 'reviews_labeled.csv'
    outfile = data_directory + 'reviews_classified_bow.csv'
    bow_file = '../Output/classification_bow.txt'
    bow_logistic_file = '../Output/logistic_reg_coefs_bow.txt'

    TEST_PROPORTION = 0.3
    TOPICS = ['service', 'food', 'money']

    df_small = pd.read_csv(infile, nrows = 100)
    cols = [c for c in df_small.columns if 'hours_' not in c]
    cols = [c for c in cols if 'attributes_' not in c]
    df = pd.read_csv(infile, usecols = cols)
    labeled_reviews = pd.read_csv(labeled_file)
    labeled_reviews = labeled_reviews.dropna(subset = TOPICS)
    labeled_reviews = labeled_reviews.drop('text', axis = 1)
    df = df.merge(labeled_reviews, on = ['review_id'], how = 'left')

    df = bag_of_words(labeled_df, df, bow_file, bow_logistic_file, TOPICS, TEST_PROPORTION)
    df.to_csv(outfile, index=False)

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

    with open(logistic_file, 'w') as outf:
        outf.write('BOW Odds ratio (exponentiated) coefficients via logistic regression\n\n')

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

def standardize_array(arr):
    means = np.mean(arr, 0)
    stds = np.std(arr, 0)
    new_arr = (arr - means) / stds
    new_arr = np.nan_to_num(new_arr)
    return new_arr

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

if __name__ == '__main__':
    main()