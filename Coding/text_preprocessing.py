import logging
import os
import re
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

logger = logging.getLogger("textpreprocessing")


def run_text_preprocessing(df):
    logger.info("===================>>> Starting Optimized Text Preprocessing (V13)")

    # 1. Targeted Noise Removal (Mentions & Whitespaces Only)
    logger.info("Normalizing Airline Mentions and cleaning whitespaces...")

    def clean_noise(text):
        if not isinstance(text, str):
            return ""

        # Replace Airline Mentions (e.g., @united) with a generic token [AIRLINE]
        # This prevents the DistilBERT model from learning a bias towards specific airlines
        text = re.sub(r"@\w+", "[AIRLINE]", text)

        # Collapse multiple spaces or tabs into a single space
        text = re.sub(r"\s+", " ", text).strip()
        return text

    df["cleaned_text"] = df["text"].apply(clean_noise)

    # Note: Lowercasing, Stemming, and Stopwords removal remain disabled
    # because DistilBERT-base-uncased captures contextual semantics from them.

    # 2. Visualization (WordCloud with plt.show)
    logger.info("\nGenerating Visualizations...")

    for sentiment, color, title in [("positive", "white", "Positive"), ("negative", "black", "Negative")]:
        logger.info(f"Generating WordCloud for {title} Tweets...")

        # Safe filtering based on sentiment column
        sentiment_mask = df["airline_sentiment"].str.lower() == sentiment
        text_data = " ".join(df[sentiment_mask]["cleaned_text"])

        if text_data.strip():
            wc = WordCloud(width=800, height=400, background_color=color).generate(text_data)
            plt.figure(figsize=(10, 5))
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")
            plt.title(f"Most Common Words in {title} Tweets")
            plt.tight_layout()
            
            # Kept as requested: Displays the WordCloud window interactively
            plt.show()

    # Calculate vocabulary reduction (for tracking purposes only)
    original_vocab = len(set(" ".join(df["text"]).split()))
    cleaned_vocab = len(set(" ".join(df["cleaned_text"]).split()))
    logger.info(f"Vocabulary Insight: Original {original_vocab} vs Cleaned {cleaned_vocab}")
    logger.info("===================>>> Optimized Preprocessing Completed!")

    return df
