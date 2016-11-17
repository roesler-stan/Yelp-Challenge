import pandas as pd
import sys
import nltk

def main():
	# infile = '../Data/reviews_clean.csv'
    infile = '../Data/reviews_classified.csv'
    outfile = '../Output/describe.txt'
	df = pd.read_csv(infile)

    orig_stdout = sys.stdout
    f = file(outfile, 'w')
    sys.stdout = f

	describe(df)
    # describe_nlp(df['text'])

    sys.stdout = orig_stdout
    f.close()

def describe(df):
    # pd.options.display.max_colwidth = 1000
    print 'Incorrect predictions\n'
    outcomes = ['food', 'service', 'money']
    for outcome in outcomes:
        print outcome + '\n'

        print 'Actual outcome, logistic regression\n'
        wrong = df[df[outcome] != df[outcome + '_pred_logit']]
        print wrong[outcome].value_counts()
        print '\n'

        print 'Actual outcome, SVM\n'
        wrong = df[df[outcome] != df[outcome + '_pred_svm']]
        print wrong[outcome].value_counts()
        print '\n'

    # service_reviews1 = df[df['service_present'] == 1]
    # service_reviews2 = df[df['service_count'] >= 2]
    # service_reviews3 = df[df['service_pred_logit'] == 1]
    # service_reviews4 = df[df['service_pred_prob_logit'] > 0.75]

    df['date'].dt.year.value_counts()

    df[:2]
    df.columns.unique()
    [c for c in df.columns if 'business' in c]

def describe_nlp(col):
    text = col.to_string()
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    # Lemmatization looks up words in a dictionary and changes to their root word
    wnl = nltk.WordNetLemmatizer()
    tokens = [wnl.lemmatize(t) for t in tokens]
    text = nltk.Text(tokens)

    print "Display all chunks with word 'service'\n"
    print text.concordance('service')
    print '\n'

    print "Display words that appear in a similar range of contexts as 'service'\n"
    print text.similar('service')
    print '\n'

    print "Display the contexts shared by 'service' and 'good'\n"
    text.common_contexts(['service', 'good'])
    print '\n'

    print "Word frequency\n"
    nltk.FreqDist(text)
    print '\n'

if __name__ == "__main__":
	main()