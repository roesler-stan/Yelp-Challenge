#!/bin/bash

# Use Yelp's script to convert JSON files to CSV
python json_to_csv_converter.py

# Make reviews_clean.csv
python clean.py

# Hand-label reviews

# Classify topics, make reviews_classified.csv
python classify.py

# Make venn diagram plots
python venn.py

# Make network edges
python edges.py

# Make network plots
python plot_nx.py
