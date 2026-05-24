# MLOps: Telecommunications Customer Churn Prediction

An end-to-end Machine Learning pipeline and serving architecture for predicting customer churn, built adhering to production-grade MLOps practices. 

## Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture and Data Flow](#system-architecture-and-data-flow)
3. [Technology Stack](#technology-stack)
4. [Course Modules Mapping](#course-modules-mapping)
5. [Repository Structure](#repository-structure)
6. [Local Setup and Execution](#local-setup-and-execution)
7. [API Documentation](#api-documentation)
8. [Monitoring and Automated Retraining](#monitoring-and-automated-retraining)

## Project Overview
This project demonstrates a complete, highly scalable machine learning lifecycle. The core objective is to predict whether a telecommunications customer is likely to cancel their subscription (churn) based on historical account metrics and demographic data. 

Beyond simply training a model, this repository showcases a robust infrastructure that handles automated testing, containerized deployment, cloud storage integration, experiment tracking, and a self-healing closed-loop monitoring system capable of triggering its own retraining pipelines when data drift is detected.

## System Architecture and Data Flow
The system is designed as a continuous, automated loop rather than a static script. The architecture follows this data flow:

1. Data Ingestion: Raw datasets are stored securely in AWS S3 and pulled dynamically into the environment.
2. Preprocessing: Automated pipelines clean, encode, and scale features.
3. Experimentation: A Random Forest classifier is trained while MLflow tracks all hyperparameters, performance metrics (Accuracy, F1 Score), and model artifacts.
4. Serving: The best-performing model is wrapped in a FastAPI REST application.
5. Containerization: The API is packaged into a Docker container to ensure absolute environmental parity across environments.
6. Orchestration: Kubernetes manifests define deployments, services, and load balancers to serve the containerized API with zero downtime.
7. Monitoring: Evidently AI evaluates simulated live traffic against baseline training data to detect feature drift.
8. Automated Retraining: If drift exceeds predefined thresholds, a Python script fires a secure webhook via GitHub Actions (repository_dispatch) to automatically trigger a pipeline execution, retraining the model on fresh data without human intervention.

## Technology Stack
* Language: Python 3.11+
* Machine Learning: Scikit-learn, Pandas, NumPy
* Experiment Tracking: MLflow
* API Framework: FastAPI, Uvicorn
* Quality Assurance: PyTest
* Containerization: Docker
* Orchestration: Kubernetes
* Monitoring: Evidently AI (v0.7+)
* CI/CD and Automation: GitHub Actions
* Cloud Storage: AWS S3

## Explanation

* Implemented a fully versioned Git repository with a modular architecture, isolating data preprocessing from model training. Reproducibility is guaranteed via explicit dependency management and automated PyTest suites.
* Developed robust feature engineering pipelines. Integrated MLflow to locally track experiments, log hyperparameter tuning iterations, and handle model versioning.
* Engineered a high-performance REST API using FastAPI. Configured GitHub Actions to create a CI/CD pipeline that automatically tests and validates the codebase on every repository push.
* Wrote Dockerfiles for lightweight containerization. Drafted Kubernetes deployment manifests (YAML) to manage self-healing replica sets and scalable infrastructure. Integrated AWS S3 for remote data storage.
* Integrated Evidently AI to generate comprehensive HTML data drift dashboards. Authored a custom monitoring script that programmatically evaluates the drift metrics and securely triggers an automated GitHub Actions retraining workflow when necessary.

## Repository Structure

    ├── .github/workflows/   # CI/CD pipeline automation (GitHub Actions)
    ├── api/                 # FastAPI application, routing, and Pydantic schemas
    ├── data/                # Local data storage (Ignored by Git, pulled from S3)
    │   ├── processed/       # Train/Test splits and scaled features
    │   └── raw/             # Raw CSV data from AWS
    ├── k8s/                 # Kubernetes deployment and service manifests (IaC)
    ├── mlruns/              # Local MLflow experiment tracking database
    ├── reports/             # Evidently AI drift detection HTML dashboards
    ├── src/                 # Core machine learning code
    │   ├── data/            # Preprocessing and feature engineering scripts
    │   └── models/          # Model training, MLflow integration, and monitoring
    ├── tests/               # PyTest suite for automated quality assurance
    ├── Dockerfile           # Blueprint for the containerized API environment
    └── requirements.txt     # Explicit Python dependencies

## Local Setup and Execution

1. Clone the repository and configure the virtual environment:
    ```bash
    git clone [https://github.com/YOUR-USERNAME/churn_prediction_mlops.git](https://github.com/YOUR-USERNAME/churn_prediction_mlops.git)
    cd churn_prediction_mlops
    python -m venv .venv
    source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

2. Download the Dataset from AWS S3:
    Note: You must have the AWS CLI installed and configured with valid IAM credentials.
    ```bash
    mkdir -p data/raw
    aws s3 cp s3://YOUR-BUCKET-NAME/data/raw/churn_data.csv data/raw/churn_data.csv
    ```

3. Execute the Machine Learning Pipeline:
    ```bash
    python src/data/preprocess.py
    python src/models/train.py
    ```

4. View the MLflow Experiment Tracking Dashboard:
    ```bash
    mlflow ui
    ```
    Navigate to `http://127.0.0.1:5000` in your web browser.

5. Start the Live FastAPI Inference Server:
    ```bash
    uvicorn api.main:app --reload
    ```
    Navigate to `http://127.0.0.1:8000/docs` to interact with the Swagger UI.

## API Documentation
The FastAPI server provides a robust `/predict` endpoint that accepts POST requests containing customer data.

Example cURL Request:
```bash
curl -X 'POST' \
  '[http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "tenure": 24,
  "MonthlyCharges": 65.50,
  "TotalCharges": 1572.00,
  "Contract_One_year": 1,
  "Contract_Two_year": 0,
  "InternetService_Fiber_optic": 0,
  "PaymentMethod_Electronic_check": 1
}'
```
Example JSON Response:
```json
{
  "prediction": "No Churn",
  "probability": 0.18
}
```
## Monitoring and Automated Retraining
To ensure the model remains accurate in production, the system utilizes a closed-loop monitoring script powered by Evidently AI.

To generate a data drift report and evaluate model health, run:
``` bash
python src/models/monitor.py
```

Process Flow:
* The script loads the baseline training data and compares it against a simulated production dataset.
* Evidently AI generates a comprehensive statistical HTML report saved in the reports/ directory.
* The script extracts the core metrics from the Evidently analysis.
* If the dataset_drift boolean evaluates to True, the script reads a secure GitHub Personal Access Token (PAT) from the local environment variables.
* A REST API call is fired to the GitHub Actions dispatches endpoint, triggering the CI/CD pipeline to pull fresh data, retrain the model, and redeploy the Kubernetes cluster entirely autonomously.
