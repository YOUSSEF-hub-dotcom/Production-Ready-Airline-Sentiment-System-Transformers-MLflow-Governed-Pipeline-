# main.py
import argparse
import logging
import os
from data_pipeline import run_data_pipeline
from text_preprocessing import run_text_preprocessing
from eda_visualization import run_eda_visualization
from model import build_and_train_distilbert_model
from MLflow_lifeCycle import run_mlflow_full_lifecycle

# 1. INITIALIZE GLOBAL LOGGING
from logger_config import setup_logging
setup_logging()
logger = logging.getLogger("MainOrchestrator")

def main():
    """
    Orchestrates the Airline Sentiment Analysis Pipeline.
    Steps: Data Ingestion -> Text Preprocessing -> EDA -> DistilBERT Training -> MLflow Registry.
    """
    # --- PHASE 1: HYPERPARAMETER PARSING ---
    parser = argparse.ArgumentParser(description="Airline Sentiment Analysis Pipeline")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate for DistilBERT")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("STARTING: AIRLINE TWEETS SENTIMENT ANALYSIS PROJECT")
    logger.info("=" * 70)

    # Dataset Source Path
    FILE_PATH = r"C:\Users\Hedaya_city\Downloads\Tweets.csv"
    
    # Check if the file exists before starting the heavy pipeline
    if not os.path.exists(FILE_PATH):
        logger.error(f"Critical Error: Dataset file not found at {FILE_PATH}")
        return

    try:
        # --- PHASE 2: DATA INGESTION & CLEANING ---
        logger.info("\n>>> STEP 1: LOADING AND INITIAL CLEANING")
        df = run_data_pipeline(FILE_PATH)

        # --- PHASE 3: NLP PREPROCESSING ---
        logger.info("\n>>> STEP 2: ADVANCED TEXT PREPROCESSING")
        df = run_text_preprocessing(df)

        # --- PHASE 4: EDA & INSIGHT GENERATION ---
        logger.info("\n>>> STEP 3: EXPLORATORY DATA ANALYSIS (EDA)")
        # Generating visualizations and statistical reports
        df = run_eda_visualization(df)

        # --- PHASE 5: TRANSFORMER MODELING (DistilBERT) ---
        logger.info(f"\n>>> STEP 4: TRAINING DistilBERT (Epochs={args.epochs}, LR={args.lr})")
        # Fine-tuning DistilBERT for multiclass sentiment classification
        model_results = build_and_train_distilbert_model(
            df,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size
        )

        # --- PHASE 6: MLOPS GOVERNANCE (MLflow) ---
        logger.info("\n>>> STEP 5: LOGGING TO MLFLOW (MODEL, METRICS & ARTIFACTS)")
        # Recording the entire run in MLflow for version control and experiment tracking
        run_mlflow_full_lifecycle(model_results)

        logger.info("=" * 70)
        logger.info(" PROJECT COMPLETED SUCCESSFULLY!")
        logger.info("=" * 70)

    except Exception as e:
        logger.critical(f"Pipeline Execution Failed: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
