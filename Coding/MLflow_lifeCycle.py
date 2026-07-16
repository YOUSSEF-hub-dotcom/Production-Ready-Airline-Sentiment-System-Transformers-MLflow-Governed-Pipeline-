import json
import logging
import mlflow
import mlflow.pyfunc
import pandas as pd
import torch
import pickle  # Added for SHAP values serialization
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.metrics import classification_report

# Initialize logger for MLflow tracking
logger = logging.getLogger("MLflow")

# Define experiment and model naming constants
EXPERIMENT_NAME = "DistilBERT-Sentiment-Analysis"
RUN_NAME = "DistilBERT_3Class_Sentiment"
MODEL_NAME = "DistilBERT_Airline_Sentiment"
ARTIFACT_PATH = "sentiment_model"


def run_mlflow_full_lifecycle(results):
    print("========== MLflow Full Lifecycle Started ==========")

    # Extract results from the training dictionary
    model = results["model"]
    tokenizer = results["tokenizer"]
    test_accuracy = results["test_accuracy"]
    preds = results["test_predictions"]
    labels = results["test_labels"]
    device = results["device"]
    params = results["params"]

    # Local paths for temporary artifact storage
    model_path = "distilbert_model"
    tokenizer_path = "distilbert_tokenizer"

    # Save model and tokenizer locally before logging to MLflow
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(tokenizer_path)

    # Set the active experiment
    mlflow.set_experiment(EXPERIMENT_NAME)

    # TRACKING: Start the MLflow run to log parameters, metrics, and artifacts
    with mlflow.start_run(run_name=RUN_NAME) as run:
        run_id = run.info.run_id
        print(f"Tracking Run ID: {run_id}")

        # Log base model info and training device
        mlflow.log_param("model", "distilbert-base-uncased")
        mlflow.log_param("device", str(device))
        
        # Log hyperparameters used during training
        mlflow.log_params(params)

        # Log primary evaluation metric
        mlflow.log_metric("test_accuracy", test_accuracy)

        # Generate and log detailed classification metrics
        report = classification_report(
            labels,
            preds,
            target_names=["negative", "neutral", "positive"],
            output_dict=True
        )

        mlflow.log_metric("macro_f1", report["macro avg"]["f1-score"])
        mlflow.log_metric("weighted_f1", report["weighted avg"]["f1-score"])

        # Save the full classification report as a JSON artifact
        with open("classification_report.json", "w") as f:
            json.dump(report, f, indent=4)

        mlflow.log_artifact("classification_report.json")

        # --- SHAP INTEGRATION: Log computed SHAP values as an MLflow run artifact ---
        if "shap_values" in results and results["shap_values"] is not None:
            logger.info("Serializing and logging SHAP values artifact...")
            shap_path = "shap_values.pkl"
            with open(shap_path, "wb") as f:
                pickle.dump(results["shap_values"], f)
            mlflow.log_artifact(shap_path)
            logger.info("SHAP values successfully logged to MLflow artifacts!")

        # PYFUNC MODEL & SIGNATURE: Define a custom Python model wrapper for inference
        class DistilBertPyFunc(mlflow.pyfunc.PythonModel):
            def load_context(self, context):
                """Load the model and tokenizer from the logged artifacts."""
                from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.tokenizer = DistilBertTokenizerFast.from_pretrained(context.artifacts["tokenizer"])
                self.model = DistilBertForSequenceClassification.from_pretrained(
                    context.artifacts["model"]
                ).to(self.device)
                self.model.eval()
                logger.info(f"PyFunc Model loaded on: {self.device}")

            def predict(self, context, model_input):
                """Process input text and return model predictions as a DataFrame."""
                texts = model_input["text"].tolist() if isinstance(model_input, pd.DataFrame) else model_input["text"]
                logger.info(f"Inference requested for {len(model_input)} samples.")

                # REFACTOR: Aligned with the new model architecture by utilizing Dynamic Padding (padding=True) 
                # inside the production PyFunc pipeline without forcing a fixed short sequence truncation.
                enc = self.tokenizer(
                    texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
                ).to(self.device)

                with torch.no_grad():
                    logits = self.model(**enc).logits
                    probs = torch.softmax(logits, dim=1)
                    preds = logits.argmax(dim=1)

                label_map = {0: "negative", 1: "neutral", 2: "positive"}

                logger.info("Inference completed successfully.")

                return pd.DataFrame({
                    "label": [label_map[p.item()] for p in preds],
                    "confidence": [probs[i][preds[i]].item() for i in range(len(preds))]
                })

        # Define input example for model signature inference
        input_example = pd.DataFrame({"text": ["The flight was delayed", "Excellent service!"]})

        # Generate dummy outputs to infer the signature schema using identical pipeline logic
        model.eval()
        with torch.no_grad():
            dummy_enc = tokenizer(input_example["text"].tolist(), padding=True, truncation=True,
                                  max_length=512, return_tensors="pt").to(device)
            dummy_logits = model(**dummy_enc).logits
            dummy_probs = torch.softmax(dummy_logits, dim=1)
            dummy_preds = dummy_logits.argmax(dim=1)

        label_map = {0: "negative", 1: "neutral", 2: "positive"}
        output_example = pd.DataFrame({
            "label": [label_map[p.item()] for p in dummy_preds],
            "confidence": [dummy_probs[i][dummy_preds[i]].item() for i in range(len(dummy_preds))]
        })

        signature = infer_signature(input_example, output_example)

        # Log the custom PyFunc model with its requirements and signature
        mlflow.pyfunc.log_model(
            artifact_path=ARTIFACT_PATH,
            python_model=DistilBertPyFunc(),
            artifacts={"model": model_path, "tokenizer": tokenizer_path},
            input_example=input_example,
            signature=signature,
            # REFACTOR: Explicitly added core dependencies to guarantee reproducible deployment isolation
            pip_requirements=["torch", "transformers", "pandas", "numpy", "scikit-learn"]
        )

    # MODEL REGISTRY: Register the logged model in the MLflow Model Registry
    client = MlflowClient()
    model_uri = f"runs:/{run_id}/{ARTIFACT_PATH}"
    print(f"Registering model from: {model_uri}")

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME
    )

    model_version = registered_model.version

    # TRANSITION → STAGING: Initially move the registered model to the Staging area
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=model_version,
        stage="Staging"
    )

    # QUALITY GATE & PRODUCTION: Check if performance meets the required thresholds
    MIN_ACCURACY = 0.75
    MIN_MACRO_F1 = 0.70

    current_macro_f1 = report["macro avg"]["f1-score"]
    status = "Staging (Low Performance)"

    # Transition to Production only if accuracy and F1-score criteria are met
    if test_accuracy >= MIN_ACCURACY and current_macro_f1 >= MIN_MACRO_F1:
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=model_version,
            stage="Production",
            archive_existing_versions=True
        )
        logger.info(f"Model Promoted! Acc: {test_accuracy:.2f}, F1: {current_macro_f1:.2f}")

        status = "Production"
    else:
        logger.warning(f"Gate Failed! Required Acc > {MIN_ACCURACY}, got {test_accuracy:.2f}")

    # Log the final summary of the model registration process
    logger.info(f"\n{'═' * 50}")
    logger.info(f" Model: {MODEL_NAME}")
    logger.info(f" Version: {model_version}")
    logger.info(f" Final Stage: {status}")
    logger.info(f"{'═' * 50}")

    return run_id
