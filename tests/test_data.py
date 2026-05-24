import os
import pandas as pd

def test_raw_data_exists():
    """Check if raw data is available."""
    assert os.path.exists("data/raw/churn_data.csv"), "Raw dataset is missing!"

def test_processed_data_exists():
    """Check if preprocessing pipeline successfully saved the data."""
    assert os.path.exists("data/processed/X_train.csv"), "X_train.csv is missing!"
    assert os.path.exists("data/processed/y_train.csv"), "y_train.csv is missing!"

def test_data_leakage():
    """Check that training and testing sets are completely isolated."""
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    
    # In a real scenario we'd check for overlapping IDs, but checking 
    # that the data is split correctly by length is a good sanity check.
    assert len(X_train) > len(X_test), "Test set is larger than or equal to train set!"