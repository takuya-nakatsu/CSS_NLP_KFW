# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 22:47:59 2023

@author: Takuya Nakatsu
""" 

# Import necessary packages to read the KFW project dataset
import json
import pandas as pd

# Load json file
with open("projects.json", encoding="utf-8") as file:
    data = json.load(file)

# Convert to pandas DataFrame
df = pd.json_normalize(data)

# Show the first few rows of the DataFrame
df.head()

# =============================================================================
# The dataset includes several features, including:
# projnr: The project number
# title: The title of the project
# description: A brief description of the project
# amount: The financial amount allocated to the project
# country: The country in which the project is based
# focus: The main focus of the project
# currency: The currency in which the amount is denominated
# responsible: The responsible authority for the project
# principal: The principal authority for the project
# crscode2: CRS code level 2 for the project
# crscode5: CRS code level 5 for the project
# finanzierungsinstrument: The financing instrument for the project
# status: The status of the project
# hostDate: The date on which the project was hosted
# fzRegion: The region for financial cooperation
# projekttraegers: The project carrier
# kofinanzpartners: Co-financing partners for the project
# usvkategorie: USV category for the project
# usvbeschr: USV description for the project
# =============================================================================
import nltk
nltk.download('omw-1.4')
# Import necessary packages for preprocessing
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
stop_words = set(stopwords.words(['german', 'english'])) #cannot change it to German

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
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words = "english") #cannot change it to German

# Vectorize the cleaned description
data_vectorized = vectorizer.fit_transform(df['description_clean'])


# =============================================================================
# LDA Topic Modeling
# =============================================================================
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
display_topics(lda_model, vectorizer.get_feature_names_out(), no_top_words)

# Get topic distribution for each document
topic_distribution = lda_model.transform(data_vectorized)

# Get the most relevant topic for each document
df['topic'] = topic_distribution.argmax(axis=1) + 1

df[['description_clean', 'topic']].head()

import matplotlib.pyplot as plt

def barplot_topics(model, feature_names, no_top_words):
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 10), sharex=True)
    topics = model.components_

    colors = ['red', 'blue', 'green', 'orange', 'purple', 'pink', 'brown', 'gray', 'yellow', 'cyan']  # Add more colors if needed

    for topic_idx, topic in enumerate(topics):
        top_words_idx = topic.argsort()[:-no_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        top_scores = [topic[i] for i in top_words_idx]

        ax = axes[topic_idx // 5, topic_idx % 5]
        ax.barh(top_words, top_scores, color=colors[topic_idx])
        ax.set_title(f"Topic {topic_idx + 1}")
        ax.invert_yaxis()
        ax.tick_params(axis='y', labelsize=8)

    plt.tight_layout()
    plt.show()

no_top_words = 8
lda_topic = barplot_topics(lda_model, vectorizer.get_feature_names_out(), no_top_words)

# =============================================================================
# Applying bert topic modeling
# =============================================================================

# Installing the BERTopic library

from bertopic import BERTopic #cannot install 

# Initialize BERTopic
topic_model = BERTopic(language="german", calculate_probabilities=True)

# Fit the model to the data
topics, probs = topic_model.fit_transform(df['description_clean'])

# Get the most frequent topics
topic_freq = topic_model.get_topic_info()
topic_freq.head()

topic_model.visualize_heatmap()
topic_model.visualize_barchart(n_words= 8, top_n_topics=10)

# =============================================================================
# Guided or seeded topic model
# https://maartengr.github.io/BERTopic/getting_started/guided/guided.html
# =============================================================================
# 
from bertopic import BERTopic




seed_topic_list = [["drug", "cancer", "drugs", "doctor"],
                   ["windows", "drive", "dos", "file"],
                   ["space", "launch", "orbit", "lunar"]]

topic_model = BERTopic(seed_topic_list=seed_topic_list)
topics, probs = topic_model.fit_transform(df['description_clean'])

# After generating topics and their probabilities, we can access the frequent topics that were generated:

topic_model.get_topic_info()

topic_model.visualize_barchart(n_words= 8, top_n_topics=10)
    
topic_model.get_topic(0)
topic_model.get_document_info(df['description_clean'])



# =============================================================================
# Comparing the project status
# =============================================================================
# Split the dataset based on project status
active_projects = df[df['status'] == 'aktiv']
completed_projects = df[df['status'] == 'abgeschlossen']


# Vectorize the cleaned descriptions
data_vectorized_active = vectorizer.fit_transform(active_projects['description_clean'])
data_vectorized_completed = vectorizer.fit_transform(completed_projects['description_clean'])

# Initialize and fit LDA Model with 10 topics for both subsets
lda_model_active = LatentDirichletAllocation(n_components=10, random_state=0)
lda_model_active.fit(data_vectorized_active)

lda_model_completed = LatentDirichletAllocation(n_components=10, random_state=0)
lda_model_completed.fit(data_vectorized_completed)

# Display the topics for active projects
display_topics(lda_model_active, vectorizer.get_feature_names_out(), no_top_words)

# Display the topics for completed projects
display_topics(lda_model_completed,  vectorizer.get_feature_names_out(), no_top_words)

# =============================================================================
# Bert Topic and Project Status
# =============================================================================
# Initialize BERTopic
topic_model = BERTopic(language="german", calculate_probabilities=True, nr_topics=10)

# Apply BERTopic to completed projects
completed_topics = topic_model.fit_transform(completed_projects['description_clean'])

# Visualize the heatmap of topic-word probabilities for completed projects
topic_model.visualize_heatmap(topics=completed_topics)

# Visualize the bar chart of the top words for each topic for completed projects
topic_model.visualize_barchart(topics=completed_topics, n_words=8, top_n_topics=10)


# Initialize BERTopic
topic_model = BERTopic(language="german", calculate_probabilities=True, nr_topics=10)

# Apply BERTopic to active projects
active_topics = topic_model.fit_transform(active_projects['description_clean'])
# Visualize the heatmap of topic-word probabilities for active projects
topic_model.visualize_heatmap(topics=active_topics)
# Visualize the bar chart of the top words for each topic for active projects
topic_model.visualize_barchart(topics=active_topics, n_words=8, top_n_topics=10)

# =============================================================================
# Translating project description into English
# =============================================================================

from googletrans import Translator

# Initialize the translator
translator = Translator()

# Example text in German
text = "Das ist ein Beispielsatz."

# Translate the text to English
translation = translator.translate(text, dest='en')

# Print the translated text
print(translation.text)

