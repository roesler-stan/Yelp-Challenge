import pandas as pd
from pandas.io.json import json_normalize
import json

def read_json(filename):
	""" Read json files, formatting them nicely for pandas """
	# read the entire file into a python array
	with open(filename, 'rb') as f:
	    data = f.readlines()

	# remove the trailing "\n" from each line
	data = map(lambda x: x.rstrip(), data)

	""" Each element of 'data' is an individual JSON object.
	I want to convert it into an *array* of JSON objects
	which, in and of itself, is one large JSON object
	basically... add square brackets to the beginning
	and end, and have all the individual business JSON objects
	separated by a comma """
	data_json_str = "[" + ','.join(data) + "]"

	# Convert the string into json
	data_json = json.loads(data_json_str)

	# Now load the data into pandas, making sure to go as deep into the nested json as necessary
	# For instance, make each type of attribute its own column
	df = pd.io.json.json_normalize(data_json)
	return df