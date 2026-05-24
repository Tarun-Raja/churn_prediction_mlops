import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import mlflow
import mlflow.sklearn
import os
import joblib

def load_processed_data():
    """Loads the preprocessed data from Phase 2."""
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze() # squeeze converts df to series
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test

def train_model():
    """Trains a Random Forest model and logs the experiment to MLflow."""
    X_train, X_test, y_train, y_test = load_processed_data()

    # Define hyperparameters
    n_estimators = 100
    max_depth = 10
    random_state = 42

    # FIX: Explicitly set a local relative file path for MLflow to avoid the Windows space bug
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
    # Set up MLflow experiment
    mlflow.set_experiment("Telco_Churn_Prediction")

    with mlflow.start_run():
        print("Training Random Forest model...")
        
        # Initialize and train the model
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=random_state
        )
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("\nClassification Report:\n", classification_report(y_test, y_pred))

        # Log parameters to MLflow
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)

        # Log metrics to MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        # Log the model to MLflow
        from mlflow.models.signature import infer_signature
        signature = infer_signature(X_train, y_pred)
        mlflow.sklearn.log_model(model, "random_forest_model", signature=signature)
        
        # Save a local copy of the model for the FastAPI server
        os.makedirs('src/models/artifacts', exist_ok=True)
        joblib.dump(model, 'src/models/artifacts/model.pkl')
        
        print("Model and metrics logged to MLflow successfully!")

if __name__ == "__main__":
    train_model()