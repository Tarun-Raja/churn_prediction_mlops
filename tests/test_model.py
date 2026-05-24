import os
import joblib
import pandas as pd

def test_model_artifacts_exist():
    """Ensure the model and scaler were saved correctly for the API."""
    assert os.path.exists("src/models/artifacts/model.pkl"), "Model artifact missing!"
    assert os.path.exists("src/models/artifacts/scaler.pkl"), "Scaler artifact missing!"
    assert os.path.exists("src/models/artifacts/features.pkl"), "Feature names missing!"

def test_model_inference():
    """Test that the saved model can successfully make a prediction on dummy data."""
    model = joblib.load("src/models/artifacts/model.pkl")
    X_test = pd.read_csv("data/processed/X_test.csv")
    
    # Grab the first row of the test set
    sample_data = X_test.iloc[[0]]
    
    # Make a prediction
    prediction = model.predict(sample_data)
    
    # The prediction should be either 0 or 1
    assert prediction[0] in [0, 1], "Model returned an invalid prediction value!"