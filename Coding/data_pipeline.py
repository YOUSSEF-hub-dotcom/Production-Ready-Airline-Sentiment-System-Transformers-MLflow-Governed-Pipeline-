import logging
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger("Data")


def load_data(file_path):
    logger.info("Loading Data .....")
    df = pd.read_csv(file_path)
    return df


def basic_data_overview(df):
    pd.set_option("display.width", None)

    print(df.head(11))

    logger.info("=======================>>> Analysis Function")
    logger.info("Information about Data:")
    print(df.info())
    logger.info("Statistical Data Analysis:")
    print(df.describe().round(2))
    logger.info("Number of rows and columns:")
    print(df.shape)
    logger.info("Columns in Data:")
    print(df.columns)
    logger.info("Frequency of Rows in Data (Duplicates):")
    print(df.duplicated().sum())


def clean_data(df):
    logger.info("=======================>>> Cleaning Data")

    logger.info("Remove Duplicates")
    df = df.drop_duplicates(keep="first").copy()
    logger.info(f"Shape after removing duplicates: {df.shape}")

    logger.info("Missing Values Before Cleaning:")
    print(df.isnull().sum())

    missing_values = df.isnull().mean() * 100
    logger.info(f"The Percentage of missing values in data:\n {missing_values.round(2)}")

    # FIX: Combined multiple strings into a single formatted string for logger.info to prevent Runtime TypeError
    logger.info(
        "The percentage of missing Value in negativereason column is: ~37% \n"
        "The percentage of missing Value in negativereason_confidence column is: ~28% \n"
        "The percentage of missing Value in tweet_location column is: ~32% \n"
        "The percentage of missing Value in user_timezone column is: ~32% \n"
        "Handling missing values using Production-safe logic..."
    )

    # FIX: Avoid filling 'negativereason' with Mode to prevent severe Data Leakage/Noise for Positive/Neutral sentiments
    # If a tweet is positive/neutral, it naturally shouldn't have a negative reason.
    df["negativereason"] = df["negativereason"].fillna("Not Applicable")
    df["negativereason_confidence"] = df["negativereason_confidence"].fillna(0.0)

    # FIX: Filling location/timezone with 'Unknown' instead of Mode to avoid introducing demographic bias
    df["tweet_location"] = df["tweet_location"].fillna("Unknown")
    df["user_timezone"] = df["user_timezone"].fillna("Unknown")

    logger.info("Missing values filled for key columns")

    # FIX: Combined multiple strings into a single formatted string for logger.info to prevent Runtime TypeError
    logger.info(
        "The Percentage of missing value in tweet_coord column is: ~93% \n"
        "The Percentage of missing value in negativereason_gold column is: ~99% \n"
        "The Percentage of missing value in airline_sentiment_gold column is: ~99% \n"
        "We will DROP these columns"
    )

    df = df.drop(columns=["tweet_coord", "negativereason_gold", "airline_sentiment_gold"], axis=1, errors="ignore")

    logger.info("Columns dropped successfully")

    logger.info("Missing Values After Full Cleaning:")
    print(df.isnull().sum())

    # FIX: Save the plot to a directory instead of using plt.show() to prevent blocking automated production pipelines
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap="viridis")
    plt.title("Missing Values in Tweets Dataset After Cleaning")
    plt.tight_layout()

    # Create docs directory if it doesn't exist to save the artifact safely
    os.makedirs("docs", exist_ok=True)
    plt.savefig("docs/missing_values_after_cleaning.png")
    plt.close()  # Properly close the figure memory
    logger.info("Missing values heatmap saved to 'docs/missing_values_after_cleaning.png'")

    return df


def run_data_pipeline(file_path):
    logger.info("============ Starting Sentiment Analysis Data Pipeline ============")

    df = load_data(file_path)

    basic_data_overview(df)

    # FIX: Explicitly assign the cleaned dataframe to ensure changes cascade down the pipeline
    df_cleaned = clean_data(df)

    logger.info(f"Final Dataset Shape: {df_cleaned.shape}")
    logger.info("Final Columns:")
    print(df_cleaned.columns.tolist())

    logger.info("============ Data Pipeline Completed Successfully! ============")

    # FIX: Changed from 'return df' to 'return df_cleaned' to guarantee that the downstream processes receive the actual cleaned dataset
    return df_cleaned
