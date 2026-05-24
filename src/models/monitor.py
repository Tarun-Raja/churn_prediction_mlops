import pandas as pd
import os
from evidently import Report
from evidently.presets import DataDriftPreset

def generate_drift_report():
    print("Loading data for drift detection...")
    
    # In a real production system, 'current_data' would be pulled from a database 
    # capturing the live traffic hitting your FastAPI endpoint. 
    reference_data = pd.read_csv("data/processed/X_train.csv")
    current_data = pd.read_csv("data/processed/X_test.csv")

    print("Generating Evidently AI Data Drift Report...")
    
    # 1. Initialize the report with the preset (New Syntax)
    report = Report([DataDriftPreset()])
    
    # 2. Run the report and capture the results (New Syntax)
    result = report.run(reference_data=reference_data, current_data=current_data)

    # 3. Save the results as an HTML dashboard (New Syntax)
    output_dir = "reports"
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "data_drift_report.html")
    
    result.save_html(report_path)
    print(f"Drift report successfully generated at: {report_path}")

if __name__ == "__main__":
    generate_drift_report()