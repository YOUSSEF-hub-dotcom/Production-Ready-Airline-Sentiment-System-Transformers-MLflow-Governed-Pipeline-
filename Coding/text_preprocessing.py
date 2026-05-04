import re
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger("textpreprocessing")

def run_text_preprocessing(df):
    logger.info("===================>>> Starting Optimized Text Preprocessing (V12)")

    # 1. Noise Removal
    # BERT requires the text as is, but URLs and mentions can confuse it
    logger.info("Removing URLs and Mentions (@user)...")

    def clean_noise(text):
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
        text = re.sub(r'@\w+', '', text)  # Remove Mentions
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
        return text

    df['cleaned_text'] = df['text'].apply(clean_noise)

    # Note: Lowercasing, Stemming, and Stopwords removal were disabled
    # because DistilBERT-base-uncased understands them and needs the context
    #

    # 2. Visualization (WordCloud)
    # Using cleaned_text to visualize the most influential words
    logger.info("\nGenerating Visualizations...")

    for sentiment, color, title in [('positive', 'white', 'Positive'), ('negative', 'black', 'Negative')]:
        logger.info(f"Generating WordCloud for {title} Tweets...")
        text_data = " ".join(df[df['airline_sentiment'] == sentiment]['cleaned_text'])

        if text_data.strip():
            wc = WordCloud(width=800, height=400, background_color=color).generate(text_data)
            plt.figure(figsize=(10, 5))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis("off")
            plt.title(f"Most Common Words in {title} Tweets")
            plt.show()

    # Calculate vocabulary reduction (for tracking purposes only)
    original_vocab = len(set(" ".join(df['text']).split()))
    cleaned_vocab = len(set(" ".join(df['cleaned_text']).split()))
    logger.info(f"Vocabulary Insight: Original {original_vocab} vs Cleaned {cleaned_vocab}")
    logger.info("===================>>> Optimized Preprocessing Completed!")

    return df


    logger.info("===================>>> Text Preprocessing Completed Successfully!")


    return df
