import pandas as pd
import nltk
import re
from nltk.stem.porter import *
import string
import datetime

def main():
    infile = '../Data/reviews_only.csv'
    df = pd.read_csv(infile, nrows = 1000)

    print 'starting to get text'
    print_time()
    raw_text = ' '.join(df['text'].astype(str).tolist()).lower()

    print 'starting clean_text'
    print_time()
    text = clean_text(raw_text)

    print 'starting get_adjectives'
    print_time()
    adjectives_text = get_adjectives(text)

    print 'done with clean_text, get_adjectives'
    print_time()

    print 'starting with select_text'
    print_time()
    text = select_text(df)

    print 'starting with parse_text'
    print_time()
    entities = parse_text(text, lemmatize = False)

    print 'starting to process adjectives'
    print_time()
    adjectives = [str(entity[0]) for entity in entities if entity[-1] == 'JJ']
    adjectives_text = ' '.join(adjectives)

    print 'done processing adjectives'
    print_time()

def print_time():
    print str(datetime.datetime.now().hour) + ':' + str(datetime.datetime.now().minute) + ':' + str(datetime.datetime.now().second)

def clean_text(text):
    text = text.decode('unicode_escape', errors = 'ignore').encode('ascii', errors = 'ignore')
    table = string.maketrans("","")
    text = text.translate(table, string.punctuation)
    return text

def get_adjectives(text):
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    # Check for named entities (e.g. "Pride and Prejudice" vs. pride) - this step takes a long time
    entities = nltk.chunk.ne_chunk(tagged)
    adjectives = [str(entity[0]) for entity in entities if entity[-1] == 'JJ']
    adjectives_text = ' '.join(adjectives)
    return adjectives_text

def select_text(df):
    """ Take df with reviews column and return text from all reviews. """
    reviews = df['text'].to_string()
    reviews = re.sub('(\d+)|([\t\n\r\f\v])','',reviews)
    while '  ' in reviews:
        reviews = reviews.replace('  ', ' ')
    return reviews

def parse_text(text, lemmatize):
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    if lemmatize:
        wnl = nltk.WordNetLemmatizer()
        tokens = [wnl.lemmatize(t) for t in tokens]

    tagged = nltk.pos_tag(tokens)
    
    # Check for named entities (e.g. "Pride and Prejudice" vs. pride) - this step takes a long time
    entities = nltk.chunk.ne_chunk(tagged)
    return entities

if __name__ == "__main__":
    main()