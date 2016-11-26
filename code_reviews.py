import re
from textblob import TextBlob
import nltk
from nltk import wordpunct_tokenize
from nltk.corpus import stopwords
import pandas as pd
import datetime

def categories(df):
    """ Code the category for ethnicities of interest and remove all other restaurants, assumes that ethnicities don't overlap """
    print 'going to code categories'
    print_time()

    ethnicities = ['Mexican', 'Italian', 'American']
    df['num_categories'] = 0
    for ethnicity in ethnicities:
        df.loc[df['categories'].str.contains(ethnicity, flags=re.I, na=False, case=False), 'category'] = ethnicity
        df.loc[df['categories'].str.contains(ethnicity, flags=re.I, na=False, case=False), 'num_categories'] += 1
    df.loc[df['category'].isnull(), 'category'] = 'Other'
    df.loc[df['num_categories'] > 1, 'category'] = 'Multiple'
    df = df[df['category'].notnull()]
    return df

def themes(df, food_file):
    """ Create column with word count for each review
    I consider "service" an explicit mention of service, not complaints about waits.
    I consider "money" an explicit mention of money, not portion sizes, happy hours, or specials
    TO DO:
        Add particular food items (e.g. "burger")
        Check for particular server names
    """
    print 'going to code themes'
    print_time()

    # Concept to look for and its regular expression
    # excluding "nice" b/c so common
    # excluding "owner" b/c often about other things, also excluding bartender b/c about drinks
    regex_dict = {'service': '(^|\W)(service|server|waiter|waitress|staff|host(ess)?\W|employee|manager|worker|busboy|' + \
    'cashier|proff?ess?ional|rude|polite|friendly|courteous|speed|fast|slow|time|wait|minute|hour|while|immediate|quick|' + \
    '(?<=mess up) order|(?<=screw up) order|order (right|wrong))',
    'speed': '(^|\W)(speed|fast|slow|time|wait|minute|hour|immediate|while|quick)',
    'good': '(^|\W)(good|great|excellent|well|amazing|fantastic|super|better)',
    'bad': '(^|\W)(bad|horribl|awful|terribl|atrocious|worse|worst|unacceptabl|outrageous)',
    'size': 'size|portion|big|small|tiny|large|huge|enormous',
    'money': '(^|\W)(cheap|expensive|deals?|bucks?|afford|spen(d|t)|charg|price|\$|dollar|money|free|cost)', # dough is ambiguous
    'food_poisoning': '(^|\W)(poison|diarr?h|diarr?ea|sick(?! of )|puke|throw up|vomit)',
    'cleanliness': '(^|\W)(clean|dirty|filthy)',
    'atmosphere': '(^|\W)(atmosphere|mood|music|loud|quiet|ambience|seating|busy)'}

    food_df = pd.read_csv(food_file, header=None, names=['word'])
    food_words = food_df['word'].tolist()
    regex_dict['food'] = '(^|\W)(food|meal|ingredients|fresh|tast|delicious|flavou?r|yum|burger|fries|' + '|'.join(food_words) + ')'

    # \S matches non-whitespace, so this counts the number of multiple non-whitespace groups: words
    df['word_count'] = df['text'].str.count('\S+', flags = re.I)
    df['characters_count'] = df['text'].str.len()

    print 'done with word and characters count'
    print_time()

    for var, regex in regex_dict.items():
        print 'starting ' + var + ' theme'
        print_time()
        df[var + '_present'] = df['text'].str.contains(regex, flags = re.I, na=False).astype(float)
        df[var + '_count'] = df['text'].str.count(regex, flags = re.I)

    print 'done with theme coding'
    print_time()

    df['polarity'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    print 'done with polarity coding'
    print_time()

    return df

def languages(df):
    def review_lang(review):
        languages_ratios = {}
        tokens = wordpunct_tokenize(review)
        words = [word.lower() for word in tokens]

        for language in stopwords.fileids():
            stopwords_set = set(stopwords.words(language))
            words_set = set(words)
            common_elements = words_set.intersection(stopwords_set)
            languages_ratios[language] = len(common_elements) # language "score"

        language = max(languages_ratios, key=languages_ratios.get)
        lang_stopwords = languages_ratios[language]
        # If there are less than 3 stop words to go on, assume it's English
        if lang_stopwords < 3:
            return 'english'

        return language

    df['language'] = df['text'].map(review_lang)

    # Only keep English-language reviews
    df = df[df['language'] == 'english']

    return df

def print_time():
    print str(datetime.datetime.now().hour) + ':' + str(datetime.datetime.now().minute) + ':' + str(datetime.datetime.now().second)