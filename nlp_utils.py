import pandas as pd
import numpy as np
import re
import string
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def clean_text(text):
    """
    Cleans raw text data for NLP analysis.
    
    Steps:
    1. Converts to lowercase.
    2. Removes URLs (http/https/www).
    3. Removes social media handles (@mentions).
    4. Removes hashtags (#).
    5. Removes punctuation.
    6. Removes extra whitespace.
    
    Args:
        text (str): The raw comment text.
        
    Returns:
        str: The cleaned text.
    """
    if pd.isna(text):
        return ""
    text = str(text).lower() # Ensure string and convert to lowercase
    text = re.sub(r"http\S+|www\S+", " ", text)  # remove URLs
    text = re.sub(r"@\w+", " ", text)           # remove @mentions
    text = re.sub(r"#\w+", " ", text)           # remove hashtags
    text = text.translate(str.maketrans("", "", string.punctuation)) # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()    # remove extra whitespace
    return text

def get_sentiment(text):
    """
    Analyzes the sentiment of a given text using TextBlob.
    
    Args:
        text (str): The cleaned text to analyze.
        
    Returns:
        tuple: (polarity_score, sentiment_label)
            - polarity_score (float): Value between -1.0 (Negative) and 1.0 (Positive).
            - sentiment_label (str): 'Positive', 'Negative', or 'Neutral'.
    """
    if not text:
        return 0.0, "Neutral"
    
    # TextBlob provides a simple API for common NLP tasks
    blob = TextBlob(text)
    score = blob.sentiment.polarity
    
    # Define thresholds for sentiment labels
    if score > 0.1:
        label = "Positive"
    elif score < -0.1:
        label = "Negative"
    else:
        label = "Neutral"
    return score, label

def perform_topic_modeling(texts, n_topics=5):
    """
    Performs unsupervised topic modeling using TF-IDF and K-Means clustering.
    
    Args:
        texts (list): List of cleaned text strings.
        n_topics (int): Number of topics (clusters) to discover.
        
    Returns:
        tuple: (clusters, topic_keywords, kmeans_model)
            - clusters (array): Cluster labels for each input text.
            - topic_keywords (list): List of top keywords for each cluster.
            - kmeans_model (KMeans): The fitted KMeans model object.
    """
    if len(texts) == 0:
        return None, None, None

    # TF-IDF Vectorization: Converts text to numerical vectors based on word importance
    # max_features=1000: Only keep top 1000 words to save memory/time
    # stop_words='english': Remove common English words (the, is, at, etc.)
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    try:
        X = vectorizer.fit_transform(texts)
    except ValueError:
        # Handle case where texts might be empty or stop words removed everything
        return None, None, None

    # K-Means Clustering: Groups similar vectors (comments) together
    kmeans = KMeans(n_clusters=n_topics, random_state=42, n_init=10)
    kmeans.fit(X)

    clusters = kmeans.labels_
    terms = vectorizer.get_feature_names_out()

    # Extract top keywords for each topic to help identify what the topic is about
    topic_keywords = []
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    for i in range(n_topics):
        # Get top 10 words for this cluster
        top_terms = [terms[ind] for ind in order_centroids[i, :10]]
        topic_keywords.append(", ".join(top_terms))

    return clusters, topic_keywords, kmeans
