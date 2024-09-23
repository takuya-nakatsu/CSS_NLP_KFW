# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 19:03:44 2023

@author: Jim
"""

# Reading the KfW project database and using a 10-topic LDA to label the projects:

import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models


# Import necessary packages to read the KFW project dataset
import json

# Load json file
with open("projects.json", encoding="utf-8") as file:
    data = json.load(file)

# Convert to pandas DataFrame
df = pd.json_normalize(data)

# Show the first few rows of the DataFrame
df.head()

# Vorbereitung der Textdaten
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Tokenisierung und Kleinschreibung
    tokens = [token for token in tokens if token.isalpha()]  # Entfernung von Nicht-Wörtern
    tokens = [token for token in tokens if token not in stopwords.words("english")]  # Entfernung von Stoppwörtern
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]  # Lemmatisierung
    return tokens

# Vorverarbeitung der Beschreibungen
df["description_tokens"] = df["description"].apply(preprocess_text)

# Erstellen des Corpus
dictionary = corpora.Dictionary(df["description_tokens"])
corpus = [dictionary.doc2bow(tokens) for tokens in df["description_tokens"]]

# Training des LDA-Modells (Latent Dirichlet Allocation)
num_topics = 10
lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)

# Extrahieren der dominierenden Topics für jedes Dokument
df["topic"] = df["description_tokens"].apply(lambda tokens: max(lda_model[dictionary.doc2bow(tokens)], key=lambda x: x[1])[0])

# Speichern des Ergebnisses in der CSV-Datei
df.to_csv("projects_with_topics.csv", index=False)

# Häufigkeitstabelle der 10 Topics

import pandas as pd
from collections import Counter

# Laden der CSV-Datei mit den Topics
df = pd.read_csv("projects_with_topics.csv")

# Zählen der Häufigkeit der Topics
topic_counts = Counter(df["topic"])

# Ausgabe der Tabelle mit der Häufigkeit der Topics
print("Topic\tFrequency")
print("----------------")
for topic, count in topic_counts.items():
    print(f"{topic}\t{count}")

# Topic model und Häufigkeitstabelle nach "fzRegion"
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
from collections import Counter

# Laden der CSV-Datei in ein DataFrame
df = pd.read_csv("projects.csv")

# Vorbereitung der Textdaten
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Tokenisierung und Kleinschreibung
    tokens = [token for token in tokens if token.isalpha()]  # Entfernung von Nicht-Wörtern
    tokens = [token for token in tokens if token not in stopwords.words("english")]  # Entfernung von Stoppwörtern
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]  # Lemmatisierung
    return tokens

# Gruppieren nach Region und Erstellen von Topic-Modellen
region_groups = df.groupby("fzRegion")
for region, group in region_groups:
    # Vorverarbeitung der Beschreibungen
    group["description_tokens"] = group["description"].apply(preprocess_text)

    # Erstellen des Corpus
    dictionary = corpora.Dictionary(group["description_tokens"])
    corpus = [dictionary.doc2bow(tokens) for tokens in group["description_tokens"]]

    # Training des LDA-Modells (Latent Dirichlet Allocation)
    num_topics = 10
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)

    # Extrahieren der dominierenden Topics für jedes Dokument
    group["topic"] = group["description_tokens"].apply(lambda tokens: max(lda_model[dictionary.doc2bow(tokens)], key=lambda x: x[1])[0])

    # Zählen der Häufigkeit der Topics
    topic_counts = Counter(group["topic"])

    # Ausgabe der Tabelle mit der Häufigkeit der Topics für die Region
    print(f"Region: {region}")
    print("Topic\tFrequency")
    print("----------------")
    for topic, count in topic_counts.items():
        print(f"{topic}\t{count}")
    print("\n")

# Topic model und Häufigkeitstabelle nach "status"
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
from collections import Counter

# Laden der CSV-Datei in ein DataFrame
df = pd.read_csv("projects.csv")

# Vorbereitung der Textdaten
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Tokenisierung und Kleinschreibung
    tokens = [token for token in tokens if token.isalpha()]  # Entfernung von Nicht-Wörtern
    tokens = [token for token in tokens if token not in stopwords.words("english")]  # Entfernung von Stoppwörtern
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]  # Lemmatisierung
    return tokens

# Gruppieren nach Status und Erstellen von Topic-Modellen
status_groups = df.groupby("status")
for status, group in status_groups:
    # Vorverarbeitung der Beschreibungen
    group["description_tokens"] = group["description"].apply(preprocess_text)

    # Erstellen des Corpus
    dictionary = corpora.Dictionary(group["description_tokens"])
    corpus = [dictionary.doc2bow(tokens) for tokens in group["description_tokens"]]

    # Training des LDA-Modells (Latent Dirichlet Allocation)
    num_topics = 10
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)

    # Extrahieren der dominierenden Topics für jedes Dokument
    group["topic"] = group["description_tokens"].apply(lambda tokens: max(lda_model[dictionary.doc2bow(tokens)], key=lambda x: x[1])[0])

    # Zählen der Häufigkeit der Topics
    topic_counts = Counter(group["topic"])

    # Ausgabe der Tabelle mit der Häufigkeit der Topics für den Status
    print(f"Status: {status}")
    print("Topic\tFrequency")
    print("----------------")
    for topic, count in topic_counts.items():
        print(f"{topic}\t{count}")
    print("\n")
