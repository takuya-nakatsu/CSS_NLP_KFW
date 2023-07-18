# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 22:47:59 2023

@author: guten
"""

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

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Initialize CountVectorizer
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')

# Vectorize the cleaned description
data_vectorized = vectorizer.fit_transform(df['description_clean'])

# Initialize LDA Model with 15 topics
lda_model = LatentDirichletAllocation(n_components=15, random_state=0)

# Fit the model on the vectorized data
lda_model.fit(data_vectorized)

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx + 1}:")
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

no_top_words = 20
display_topics(lda_model, vectorizer.get_feature_names(), no_top_words)
