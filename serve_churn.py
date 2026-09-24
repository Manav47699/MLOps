import os
from contextlib import asynccontextmanager
from typing import Optional
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import mlflow.sklearn

MODEL_URI = os.getenv("CHURN_MODEL_URI", "models:/TelcoChurnModel/Production")
model_pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_pipeline
    try:
        print(f"Loading production model from {MODEL_URI}...")
        model_pipeline = mlflow.sklearn.load_model(MODEL_URI)
        print("Model loaded successfully into FastAPI service.")
    except Exception as e:
        print(f"Failed to load model from registry ({e}). Trying fallback models:/TelcoChurnModel/1...")
        try:
            model_pipeline = mlflow.sklearn.load_model("models:/TelcoChurnModel/1")
            print("Loaded fallback model version 1.")
        except Exception as e2:
            print(f"Warning: Could not load model: {e2}")
    yield


app = FastAPI(
    title="Telco Customer Churn Serving API",
    description="Loads best production model from MLflow Model Registry and performs real-time inference.",
    lifespan=lifespan,
)


class CustomerData(BaseModel):
    gender: str = Field(default="Female", example="Female")
    SeniorCitizen: int = Field(default=0, example=0)
    Partner: str = Field(default="Yes", example="Yes")
    Dependents: str = Field(default="No", example="No")
    tenure: int = Field(default=12, example=12)
    PhoneService: str = Field(default="Yes", example="Yes")
    MultipleLines: str = Field(default="No", example="No")
    InternetService: str = Field(default="DSL", example="DSL")
    OnlineSecurity: str = Field(default="No", example="No")
    OnlineBackup: str = Field(default="Yes", example="Yes")
    DeviceProtection: str = Field(default="No", example="No")
    TechSupport: str = Field(default="No", example="No")
    StreamingTV: str = Field(default="No", example="No")
    StreamingMovies: str = Field(default="No", example="No")
    Contract: str = Field(default="Month-to-month", example="Month-to-month")
    PaperlessBilling: str = Field(default="Yes", example="Yes")
    PaymentMethod: str = Field(default="Electronic check", example="Electronic check")
    MonthlyCharges: float = Field(default=55.20, example=55.20)
    TotalCharges: float = Field(default=662.40, example=662.40)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model_pipeline is not None,
        "model_uri": MODEL_URI,
    }


@app.post("/predict")
def predict_churn(customer: CustomerData):
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded or unavailable.")

    try:
        input_dict = customer.model_dump()
        input_df = pd.DataFrame([input_dict])

        pred = int(model_pipeline.predict(input_df)[0])
        proba = float(model_pipeline.predict_proba(input_df)[0][1])

        return {
            "churn_prediction": pred,
            "churn_label": "Yes" if pred == 1 else "No",
            "churn_probability": round(proba, 4),
            "risk_level": "High" if proba >= 0.5 else "Low",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
