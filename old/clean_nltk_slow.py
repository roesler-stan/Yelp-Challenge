import nltk
from nltk.stem.porter import *
import string

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
