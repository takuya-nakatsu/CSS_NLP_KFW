# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 22:47:59 2023

@author: guten
"""
import os

# Rename a file
os.rename("untitled1.py", "kfw.nlp.de.py")

# Rename a directory
os.rename("old_folder", "new_folder")

import json
import pandas as pd

# Load json file
with open("projects.json", encoding="utf-8") as file:
    data = json.load(file)

# Convert to pandas DataFrame
df = pd.json_normalize(data)

# Show the first few rows of the DataFrame
df.head()

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
import nltk

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Set stopwords for German and English
stop_words = set(stopwords.words(['german', 'english']))

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Remove HTML tags, URLs, digits, and punctuation
    text = re.sub('<.*?>|http\S+|[0-9]+|\W', ' ', text)
    
    # Tokenize, lemmatize, and remove stopwords
    text = [lemmatizer.lemmatize(word) for word in word_tokenize(text.lower()) if word not in stop_words]
    
    return ' '.join(text)

# Apply the preprocessing function to the 'description' column
df['description_clean'] = df['description'].apply(preprocess_text)

# Display the first few rows of the DataFrame
df[['description', 'description_clean']].head()
