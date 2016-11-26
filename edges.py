"""
Read Mugshots or NIBRS dataset and write an adjacency list or a charge-charge edge list
"""

import csv
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import scipy

def main():
	out_directory = '../Data/'
	infile = out_directory + 'reviews_classified.csv'
	edges_file = out_directory + 'edges.csv'
	food_edges = out_directory + 'edges_food.csv'
	nonfood_edges = out_directory + 'edges_nonfood.csv'
	service_edges = out_directory + 'edges_service.csv'
	nonservice_edges = out_directory + 'edges_nonservice.csv'
	money_edges = out_directory + 'edges_money.csv'
	nonmoney_edges = out_directory + 'edges_nonmoney.csv'

	df = pd.read_csv(infile, usecols = ['text', 'food_present', 'service_present', 'money_present'])
	df['text'] = df['text'].astype(str)
	reviews = df['text'].tolist()
	food_reviews = df.loc[df['food_present'] == 1, 'text'].tolist()
	nonfood_reviews = df.loc[df['food_present'] == 0, 'text'].tolist()
	service_reviews = df.loc[df['service_present'] == 1, 'text'].tolist()
	nonservice_reviews = df.loc[df['service_present'] == 0, 'text'].tolist()
	money_reviews = df.loc[df['money_present'] == 1, 'text'].tolist()
	nonmoney_reviews = df.loc[df['money_present'] == 0, 'text'].tolist()

	TOP_WORDS = 1000
	cols, comatrix = get_comatrix(reviews, TOP_WORDS)
	write_csv(cols, comatrix, edges_file)

	cols, comatrix = get_comatrix(food_reviews, TOP_WORDS)
	write_csv(cols, comatrix, food_edges)

	cols, comatrix = get_comatrix(nonfood_reviews, TOP_WORDS)
	write_csv(cols, comatrix, nonfood_edges)

	cols, comatrix = get_comatrix(service_reviews, TOP_WORDS)
	write_csv(cols, comatrix, service_edges)

	cols, comatrix = get_comatrix(nonservice_reviews, TOP_WORDS)
	write_csv(cols, comatrix, nonservice_edges)

	cols, comatrix = get_comatrix(money_reviews, TOP_WORDS)
	write_csv(cols, comatrix, money_edges)

	cols, comatrix = get_comatrix(nonmoney_reviews, TOP_WORDS)
	write_csv(cols, comatrix, nonmoney_edges)

def get_comatrix(reviews, TOP_WORDS=None, MIN_DOC_FREQ=1, MAX_DOC_FREQ=1.0):
	count_model = CountVectorizer(ngram_range=(1,1), stop_words='english', max_features=TOP_WORDS, min_df=MIN_DOC_FREQ, max_df=MAX_DOC_FREQ) # default unigram model
	X = count_model.fit_transform(reviews)
	Xc = (X.T * X) # this is co-occurrence matrix in sparse csr format
	cols = count_model.get_feature_names()
	comatrix = scipy.sparse.coo_matrix(Xc)
	return cols, comatrix

def write_csv(cols, comatrix, outfile):
	with open(outfile, 'w') as outf:
		headers = ['source', 'target', 'weight']
		writer = csv.DictWriter(outf, fieldnames=headers)
		# writer.writeheader()
		for source_no, target_no, weight in zip(comatrix.row, comatrix.col, comatrix.data):
			# Only get the upper triangle, and ignore self-edges
			if source_no >= target_no:
				continue
			source = cols[source_no]
			target = cols[target_no]
			writer.writerow({'source': source, 'target': target, 'weight': weight})

def new(SG):
	 nx.shortest_path(SG, 'food', 'service', weighted=True)

if __name__ == '__main__':
    main()