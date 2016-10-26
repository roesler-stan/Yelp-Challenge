import pandas as pd

def main():
	infile = '../Data/reviews_clean.csv'
	df = pd.read_csv(infile)

	df['stars_review'].unique()

	df.reset_index()[:4]

	df = df.drop('Unnamed: 0', axis = 1)
	df.to_csv(outfile, index = False)

if __name__ == "__main__":
	main()