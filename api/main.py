from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
from api.schemas import ChurnPredictionRequest

app = FastAPI(
    title="Customer Churn Prediction API",
    description="An API that predicts whether a Telco customer will churn based on their profile.",
    version="1.0"
)

# Load artifacts on startup to keep inference fast
try:
    model = joblib.load('src/models/artifacts/model.pkl')
    scaler = joblib.load('src/models/artifacts/scaler.pkl')
    features = joblib.load('src/models/artifacts/features.pkl')
except Exception as e:
    print(f"Error loading model artifacts: {e}. Did you run Phase 2 and 3?")

@app.get("/")
def home():
    return {"message": "Welcome to the Telco Churn Prediction API. Go to /docs to test it."}

@app.post("/predict")
def predict(request: ChurnPredictionRequest):
    try:
        # 1. Convert incoming JSON request into a Pandas DataFrame
        # request.model_dump() converts the Pydantic object to a dictionary
        input_data = pd.DataFrame([request.model_dump()])

        # 2. Separate column types (just like in training)
        categorical_cols = input_data.select_dtypes(include=['object']).columns
        
        # 3. Dummy Encode Categorical Variables
        input_encoded = pd.get_dummies(input_data, columns=categorical_cols, drop_first=False)
        
        # 4. Align columns with training features
        # If the user didn't have certain categories, this fills the missing dummy columns with 0
        input_aligned = input_encoded.reindex(columns=features, fill_value=0)
        
        # 5. Scale numerical features using the saved scaler
        num_cols = scaler.feature_names_in_
        input_aligned[num_cols] = scaler.transform(input_aligned[num_cols])
        
        # 6. Make prediction
        prediction = model.predict(input_aligned)
        probability = model.predict_proba(input_aligned)[0][1]

        return {
            "churn_prediction": "Yes" if int(prediction[0]) == 1 else "No",
            "churn_probability": float(probability)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))