import pandas as pd
import os
import requests
# The correct v0.7+ imports
from evidently import Report
from evidently.presets import DataDriftPreset

def trigger_github_actions():
    """Fires a secure API call to GitHub to start the retraining pipeline."""
    GITHUB_USERNAME = "Tarun-Raja"
    REPO_NAME = "churn_prediction_mlops"
    
    # Securely grab the token from the environment
    token = os.environ.get("GH_PAT")
    if not token:
        print("Warning: GH_PAT environment variable not set. Cannot trigger retrain.")
        return

    url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{REPO_NAME}/dispatches"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {token}"
    }
    # This must match the 'types' we added to the .yml file!
    payload = {"event_type": "retrain-model"}

    print("Firing deployment trigger to GitHub Actions...")
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 204:
        print("Success! Retraining pipeline has been triggered in the cloud.")
    else:
        print(f"Failed to trigger pipeline. Status: {response.status_code}, {response.text}")

def generate_drift_report():
    print("Loading data for drift detection...")
    reference_data = pd.read_csv("data/processed/X_train.csv")
    current_data = pd.read_csv("data/processed/X_test.csv")

    print("Generating Evidently AI Data Drift Report...")
    # Initialize Report with the correct v0.7 Preset
    report = Report([DataDriftPreset()])
    
    # The .run() method returns a Snapshot object
    result = report.run(reference_data=reference_data, current_data=current_data)

    output_dir = "reports"
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "data_drift_report.html")
    
    # Save the HTML directly from the result snapshot (This works!)
    result.save_html(report_path)
    print(f"Drift report successfully generated at: {report_path}")

    # --- THE CLOSED LOOP AUTOMATION ---
    print("Analyzing drift metrics...")
    
    # Use .dict() instead of the deprecated .as_dict()
    metrics = result.dict()
    
    # Safely extract the drift boolean
    try:
        drift_detected = metrics["metrics"][0]["result"]["dataset_drift"]
    except KeyError:
        # Safe fallback search just in case the JSON structure shifts
        drift_detected = "'dataset_drift': True" in str(metrics)

    if drift_detected:
        print("CRITICAL: Data drift threshold exceeded!")
        trigger_github_actions()
    else:
        print("No significant data drift detected. Model is healthy. No retrain needed.")
        
        # FOR TESTING ONLY: 
        # Uncomment the line below to force the trigger and prove the cloud loop works:
        #trigger_github_actions()

if __name__ == "__main__":
    generate_drift_report()