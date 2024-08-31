# logmodel.py

import mlflow
import mlflow.pytorch
from transformers import DistilBertForSequenceClassification
import torch

# Define your model path (update this path according to your setup)
model_path = r'C:\Users\CSE_BAY3\CustomerFeedbackAnalysis\fine-tuned-model'

# Load the model
model = DistilBertForSequenceClassification.from_pretrained(model_path)

# Define parameters
value1 = 0.01  # Example parameter
value2 = 64     # Example parameter

# Create or set an experiment
experiment_name = "SentimentAnalysisExperiment"
mlflow.set_experiment(experiment_name)

# Start an MLflow run
with mlflow.start_run() as run:
    # Log the Hugging Face model
    mlflow.log_artifact(model_path, "model")
    
    # Log parameters
    mlflow.log_params({
        "param1": value1,
        "param2": value2
    })

    # Print the run ID to track your model version
    print(f"Model logged with run ID: {run.info.run_id}")
