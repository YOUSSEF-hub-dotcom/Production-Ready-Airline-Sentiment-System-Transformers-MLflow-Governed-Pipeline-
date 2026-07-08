import logging
import os
import re
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from wordcloud import WordCloud

# Download required NLTK components quietly to keep the logs clean
nltk.download(['punkt', 'stopwords', 'wordnet', 'omw-1.4'], quiet=True)

# Define core configuration and application styling
st.set_page_config(page_title="✈️ Airline Sentiment Dashboard", layout="wide", page_icon="✈️")

# Logging setup for the analytical module
logger = logging.getLogger("EDA_Dashboard")


@st.cache_data
def load_data():
    """
    Loads, cleans, processes, and builds analytical features 
    from the raw airline dataset for immediate dashboard consumption.
    """
    FILE_PATH = r"C:\Users\Hedaya_city\Downloads\Tweets.csv"
    
    if not os.path.exists(FILE_PATH):
        st.error(f" Dataset file not found at: {FILE_PATH}. Please verify the path.")
        st.stop()
        
    df = pd.read_csv(FILE_PATH, encoding="latin1")
    df = df.drop_duplicates()

    # Handling missing values professionally using stable modes/medians
    df['negativereason'] = df['negativereason'].fillna(df['negativereason'].mode()[0])
    df['negativereason_confidence'] = df['negativereason_confidence'].fillna(df['negativereason_confidence'].median())
    df['tweet_location'] = df['tweet_location'].fillna(df['tweet_location'].mode()[0])
    df['user_timezone'] = df['user_timezone'].fillna(df['user_timezone'].mode()[0])

    # Dropping columns containing critical missing thresholds
    df = df.drop(columns=['tweet_coord', 'negativereason_gold', 'airline_sentiment_gold'], errors='ignore')

    # --- NLP Text Preprocessing Features Generation ---
    df['lower_text'] = df['text'].str.lower()
    df['tokenized_text'] = df['lower_text'].apply(nltk.word_tokenize)
    df['no_specials'] = df['tokenized_text'].apply(lambda x: [re.sub(r'[^a-zA-Z]', '', w) for w in x])
    stop_words = set(stopwords.words('english'))
    df['no_stopwords'] = df['no_specials'].apply(lambda x: [w for w in x if w not in stop_words and w != ''])
    
    lemmatizer = WordNetLemmatizer()
    df['lemmatized_text'] = df['no_stopwords'].apply(lambda x: [lemmatizer.lemmatize(w) for w in x])
    df['cleaned_text'] = df['lemmatized_text'].apply(lambda x: " ".join(x))

    # --- Time & Statistical Metadata Feature Extraction ---
    df['tweet_created'] = pd.to_datetime(df['tweet_created'])
    df['hour'] = df['tweet_created'].dt.hour
    df['text_length'] = df['text'].apply(len)
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))
    df['tweet_length'] = df['cleaned_text'].apply(lambda x: len(x.split()))

    return df


# Ingest data safely into application state
df = load_data()

st.title("✈️ Airline Customer Sentiment Dashboard")
st.markdown("Explore and visualize structural patterns inside customer feedback and brand mentions.")
st.markdown("---")

# REFACTOR: Switched from a limiting sidebar selectbox to an interactive modern native Tab layout
tabs = st.tabs([
    "📘 Sentiment Overview",
    "📊 Sentiment per Airline",
    "📉 Negative Feedback",
    "📈 Positive Feedback",
    "📝 Word Clouds",
    "🕒 Users & Timing",
    "📏 Tweet Metrics",
    "📌 Additional Insights"
])

# ---------------- Tab 1: Sentiment Overview ----------------
with tabs[0]:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Global Sentiment Proportions")
        sentiment_percent = round(df['airline_sentiment'].value_counts(normalize=True) * 100, 2)
        fig, ax = plt.subplots(figsize=(5, 5))
        sentiment_percent.plot.pie(autopct='%1.1f%%', ax=ax, colors=['#ff9999','#66b3ff','#99ff99'], startangle=90, wedgeprops=dict(width=0.4))
        ax.set_ylabel('')
        st.pyplot(fig)
    with col2:
        st.subheader("Volume of Volume per Airline")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.countplot(x='airline', data=df, ax=ax, palette="Blues_d")
        plt.xticks(rotation=45)
        st.pyplot(fig)

# ---------------- Tab 2: Sentiment per Airline ----------------
with tabs[1]:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Sentiment Cross-Distribution")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.countplot(x='airline', hue='airline_sentiment', data=df, ax=ax, palette="Set2")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    with col2:
        st.subheader("Cleaned Tweet Length Dynamics")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.boxplot(x='airline_sentiment', y='tweet_length', data=df, ax=ax, palette="pastel")
        st.pyplot(fig)

# ---------------- Tab 3: Negative Feedback ----------------
with tabs[2]:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Primary Driver for Complaints")
        neg_reason = df[df['airline_sentiment'] == 'negative']['negativereason'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.barplot(x=neg_reason.head(5).index, y=neg_reason.head(5).values, ax=ax, palette="Reds_r")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    with col2:
        st.subheader("Airlines with Most Negative Volume")
        neg_airline = df[df['airline_sentiment'] == 'negative']['airline'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.barplot(x=neg_airline.index, y=neg_airline.values, ax=ax, palette="dark:red_r")
        plt.xticks(rotation=45)
        st.pyplot(fig)

# ---------------- Tab 4: Positive Feedback ----------------
with tabs[3]:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Most Appreciated Operators")
        pos_airline = df[df['airline_sentiment'] == 'positive']['airline'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.barplot(x=pos_airline.index, y=pos_airline.values, ax=ax, palette="YlGnBu_r")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    with col2:
        st.subheader("Direct Contrast: Positive vs Negative")
        sentiment_airline = df.groupby(['airline', 'airline_sentiment']).size().unstack()
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sentiment_airline[['positive', 'negative']].plot(kind='bar', ax=ax, color=['#2ca02c', '#d62728'])
        plt.xticks(rotation=0)
        st.pyplot(fig)

# ---------------- Tab 5: Word Clouds ----------------
with tabs[4]:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Keywords inside Positive Interactions")
        positive_text = " ".join(df[df['airline_sentiment'] == 'positive']['cleaned_text'])
        wordcloud_pos = WordCloud(width=700, height=350, background_color='white').generate(positive_text)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.imshow(wordcloud_pos, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    with col2:
        st.subheader("Keywords inside Negative Interactions")
        negative_text = " ".join(df[df['airline_sentiment'] == 'negative']['cleaned_text'])
        wordcloud_neg = WordCloud(width=700, height=350, background_color='black', colormap="Reds").generate(negative_text)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.imshow(wordcloud_neg, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

# ---------------- Tab 6: Users & Timing ----------------
with tabs[5]:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top Influential Critics/Authors")
        top_users = df['name'].value_counts().head(5)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.barplot(x=top_users.index, y=top_users.values, ax=ax, palette="rocket")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    with col2:
        st.subheader("Geographical Distribution Hotspots")
        top_locations = df['tweet_location'].value_counts().head(5)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.barplot(x=top_locations.index, y=top_locations.values, ax=ax, palette="mako")
        plt.xticks(rotation=45)
        st.pyplot(fig)

# ---------------- Tab 7: Tweet Metrics ----------------
with tabs[6]:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Hourly Distribution Density")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.countplot(x='hour', data=df, ax=ax, palette="magma")
        st.pyplot(fig)
    with col2:
        st.subheader("Viral Impact: Retweets vs Sentiment")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.boxplot(x='airline_sentiment', y='retweet_count', data=df, ax=ax, palette="vlag")
        st.pyplot(fig)

# ---------------- Tab 8: Additional Insights ----------------
with tabs[7]:
    col1, col2 = st.columns(2)
    with col1:
        # REFACTOR: Replaced duplicate chart with actionable information regarding hourly complaints density
        st.subheader("Hourly Peak for Complaints Generation")
        negative_hours = df[df['airline_sentiment'] == 'negative']
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.countplot(x='hour', data=negative_hours, ax=ax, color="#d62728")
        st.pyplot(fig)
    with col2:
        st.subheader("Confidence Scoring Profile by Sentiment Type")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.boxplot(x='airline_sentiment', y='airline_sentiment_confidence', data=df, ax=ax, palette="deep")
        st.pyplot(fig)

# ---------------- Footer ----------------
st.divider()
st.caption("Airline Sentiment Insights Exploration Platform | Predictive Analytics Framework 2026")
