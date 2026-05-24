import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_data(filepath="data/raw/churn_data.csv"):
    """Loads the raw dataset."""
    df = pd.read_csv(filepath)
    # Drop customerID as it has no predictive power
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
    return df

def clean_data(df):
    """Handles missing values and incorrect data types."""
    # TotalCharges is sometimes loaded as an object because of blank spaces.
    # We force it to numeric and fill any resulting NaNs with 0.
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    return df

def preprocess_and_split(df, test_size=0.2, random_state=42):
    """Encodes categoricals, scales numericals, and splits the data."""
    
    # 1. Separate features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

    # 2. Identify column types
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(exclude=['object']).columns

    # 3. Encode categorical variables
    # For a simple Random Forest/Logistic Regression, dummy encoding is standard.
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # 4. Split the data FIRST to prevent data leakage during scaling
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 5. Scale numerical features based ONLY on training data
    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    # Apply the same transformation to test data without re-fitting
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    # 6. Save the scaler and feature names for inference later
    os.makedirs('src/models/artifacts', exist_ok=True)
    joblib.dump(scaler, 'src/models/artifacts/scaler.pkl')
    joblib.dump(list(X_train.columns), 'src/models/artifacts/features.pkl')

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # This block allows us to run this file directly to test it.
    print("Loading data...")
    raw_df = load_data()
    
    print("Cleaning data...")
    cleaned_df = clean_data(raw_df)
    
    print("Preprocessing and splitting data...")
    X_train, X_test, y_train, y_test = preprocess_and_split(cleaned_df)
    
    print("Saving processed data...")
    os.makedirs("data/processed", exist_ok=True)
    
    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)
    
    print(f"Pipeline complete! Training set size: {X_train.shape}")