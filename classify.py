"""
Train a Naive Bayes and a Bayesian logistic regression model, then test and use to classify unlabeled reviews
as to whether they judge the service.

To do:
	1. Scale features to be comparable
	2. Once you are satisified with a mode, test the accuracy on observations NEVER used to train or test a model
"""

import nltk
from nltk.classify import NaiveBayesClassifier
import pandas
import numpy
import sklearn

def main():
	data_directory = '../../Data/'
	infile = data_directory + 'yelp_reviews_only_training.csv'
	outfile = data_directory + 'yelp_reviews_classified.csv'
	outfile_small = data_directory + 'yelp_reviews_classified_small.csv'

	dataset = pandas.read_csv(infile)

	# dataset = normalize(dataset)

	independent_vars = ['characters_count', 'mexican_count', 'strange_count', 'authentic_count', 'tasty_present']
	dependent_var = 'discusses_authentic'

	coded_rows = 51
	training_size = int(round(coded_rows * 0.7))
	test_size = int(round(coded_rows * 0.3))

	dataset = naive_bayes(dataset, independent_vars, dependent_var, training_size, test_size)
	dataset = logistic_regression(dataset, independent_vars, dependent_var, training_size, test_size)

	dataset.to_csv(outfile, index = False)
	dataset[0:100].to_csv(outfile_small, index = False)

def naive_bayes(dataset, independent_vars, dependent_var, training_size, test_size):
	def classify_review(row_number):
		""" Predict uncoded reviews' depdendent variable """
		return classifier.classify(features_dict[row_number])

	def label_probability(row_number):
		""" Predict probability of the outcome """
		outcome = 1
		return classifier.prob_classify(features_dict[row_number]).prob(outcome)

	# Create features dictionary, with each review ID (row number) as key, and features dict as value
	features_dict = dataset[independent_vars].to_dict('review_id')
	dependent_var_dict = dataset[[dependent_var]].to_dict('review_id')

	# Create (features_dict, label) tuples, e.g. ({'word_count': 10, 'mexican_count': 2}, 1)
	featuresets = [(features_dict[row_number], dependent_var_dict[row_number][dependent_var]) for row_number in dataset.index]
	train_set, test_set = featuresets[:training_size], featuresets[training_size: training_size + test_size]

	classifier = nltk.NaiveBayesClassifier.train(train_set)

	print 'Classifier accuracy (correctly classified test examples)'
	print(nltk.classify.accuracy(classifier, test_set))

	print '\nThe most informative features:'
	classifier.show_most_informative_features()

	predicted_var = dependent_var + '_pred_nbayes'
	probability_var = dependent_var + '_prob_nbayes'
	dataset[predicted_var] = dataset.index.map(classify_review)
	dataset[probability_var] = dataset.index.map(label_probability)
	
	correctly_classified = dependent_var + '_correct_nbayes'
	dataset[correctly_classified] = (dataset[predicted_var] == dataset[dependent_var]).astype(float)
	dataset.loc[dataset[dependent_var].isnull(), correctly_classified] = numpy.nan
	dataset.loc[dataset[predicted_var].isnull(), correctly_classified] = numpy.nan

	print '\nTotal correctly classified examples via Naive Bayes:'
	print dataset[correctly_classified].mean()

	return dataset

def logistic_regression(dataset, independent_vars, dependent_var, training_size, test_size):
	training_set = dataset[dataset[dependent_var].notnull()]
	training_set = training_set[:training_size]
	for var in independent_vars:
		training_set = training_set[training_set[var].notnull()]

	# Train a logistic regression model with an L2 penalty
	# Alternative: use Bayesian logistic regression with a Bayesian prior distribution on the parameters
	logreg = sklearn.linear_model.LogisticRegression(penalty = 'l2', solver = 'sag')
	model = logreg.fit(training_set[independent_vars], training_set[dependent_var])
		
	print '\nOdds ratio (exponentiated) coefficients via logistic regression:'
	print pandas.DataFrame(zip(independent_vars, numpy.transpose(model.coef_)))

	# Predicted value
	predicted_var = dependent_var + '_pred_logit'
	dataset[predicted_var] = model.predict(dataset[independent_vars])

	# Predicted probability that dependent variable is positive
	dataset[dependent_var + '_pred_prob_logit'] = model.predict_proba(dataset[independent_vars])[:,1]

	correctly_classified = dependent_var + '_correct_logit'
	dataset[correctly_classified] = (dataset[predicted_var] == dataset[dependent_var]).astype(float)
	dataset.loc[dataset[dependent_var].isnull(), correctly_classified] = numpy.nan
	dataset.loc[dataset[predicted_var].isnull(), correctly_classified] = numpy.nan

	test_set = dataset[training_size: training_size + test_size]
	print '\nCorrectly classified test examples via logistic regression:'
	print test_set[correctly_classified].mean()

	print '\nTotal correctly classified examples via logistic regression:'
	print dataset[correctly_classified].mean()

	return dataset

def normalize(dataset):
	for col in dataset.columns:
		# subtract the mean
		col_mean = dataset[col].mean()
		dataset[col] = dataset[col] - col_mean

		# divide by standard deviation
		col_sd = dataset[col].std()
		dataset[col] = dataset[col] / col_sd

	return dataset

if __name__ == '__main__':
	main()