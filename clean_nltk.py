import nltk
from nltk.stem.porter import *
import re

def select_text(col):
    """ Take reviews column and return text from all reviews. """
    reviews = col.to_string()
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
