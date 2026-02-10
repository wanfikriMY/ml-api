import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from app.models import (
    LoanApplication,
    LoanBatch,
    LoanPredictionResponse,
    ErrorResponse,
    Iris,
)
from joblib import load
import numpy as np
import pandas as pd


LOAN_MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models",
    "loan-approval",
    "results",
    "random_forest_model.joblib",
)
LOAN_ENCODERS_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models",
    "loan-approval",
    "results",
    "label_encoders.joblib",
)
MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "iris-model", "dt_model.pkl"
)
ENCODER_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "iris-model", "label_encoder.pkl"
)


loan_model = None
loan_encoders = None


CATEGORICAL_COLUMNS = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
]
NUMERICAL_COLUMNS = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
]


def preprocess_loan_input(loan: LoanApplication):
    input_data = {
        "Gender": loan.gender,
        "Married": loan.married,
        "Dependents": loan.dependents if loan.dependents else "0",
        "Education": loan.education,
        "Self_Employed": loan.self_employed if loan.self_employed else "No",
        "ApplicantIncome": loan.applicant_income,
        "CoapplicantIncome": loan.coapplicant_income,
        "LoanAmount": loan.loan_amount,
        "Loan_Amount_Term": loan.loan_amount_term,
        "Credit_History": loan.credit_history,
        "Property_Area": loan.property_area,
    }
    df = pd.DataFrame([input_data])
    for col in CATEGORICAL_COLUMNS:
        if col in loan_encoders:
            le = loan_encoders[col]
            df[col] = le.transform(df[col].astype(str))
    return df.values


def validate_loan_input(loan: LoanApplication):
    errors = []
    if loan.applicant_income < 0:
        errors.append("Applicant income cannot be negative")
    if loan.coapplicant_income < 0:
        errors.append("Coapplicant income cannot be negative")
    if loan.loan_amount <= 0:
        errors.append("Loan amount must be positive")
    if loan.loan_amount_term <= 0:
        errors.append("Loan amount term must be positive")
    if loan.credit_history not in [0.0, 1.0]:
        errors.append("Credit history must be 0.0 or 1.0")
    if loan.gender not in ["Male", "Female"]:
        errors.append("Gender must be 'Male' or 'Female'")
    if loan.married not in ["Yes", "No"]:
        errors.append("Married must be 'Yes' or 'No'")
    if loan.education not in ["Graduate", "Not Graduate"]:
        errors.append("Education must be 'Graduate' or 'Not Graduate'")
    if loan.property_area not in ["Urban", "Rural", "Semiurban"]:
        errors.append("Property area must be 'Urban', 'Rural', or 'Semiurban'")
    if loan.dependents is not None and loan.dependents not in ["0", "1", "2", "3+"]:
        errors.append("Dependents must be '0', '1', '2', or '3+'")
    if loan.self_employed is not None and loan.self_employed not in ["Yes", "No"]:
        errors.append("Self employed must be 'Yes' or 'No'")
    return errors


@asynccontextmanager
async def lifespan(app: FastAPI):
    global loan_model, loan_encoders
    app.state.iris_model = load(MODEL_PATH)
    app.state.iris_encoder = load(ENCODER_PATH)
    app.state.class_names = app.state.iris_encoder.classes_.tolist()
    loan_model = load(LOAN_MODEL_PATH)
    loan_encoders = load(LOAN_ENCODERS_PATH)
    yield


app = FastAPI(
    title="Iris ML API",
    description="API for ml model",
    version="1.0",
    lifespan=lifespan,
)


@app.get("/health", tags=["health"])
async def health_check():
    return {"status": "healthy"}


@app.post("/iris/predict", tags=["predictions"])
async def get_prediction(iris: Iris):
    data = iris.data
    class_names = app.state.class_names
    iris_model = app.state.iris_model
    prediction = iris_model.predict(data).tolist()
    proba = iris_model.predict_proba(data).tolist()
    prediction_name = [class_names[int(p)] for p in prediction]
    return {
        "prediction": prediction,
        "prediction_name": prediction_name,
        "proba": proba,
    }


@app.post("/loan/predict", response_model=LoanPredictionResponse, tags=["predictions"])
async def predict_loan(loan: LoanApplication):
    validation_errors = validate_loan_input(loan)
    if validation_errors:
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "INVALID_INPUT",
                "message": "Input validation failed",
                "details": {"errors": validation_errors},
            },
        )

    try:
        features = preprocess_loan_input(loan)
        prediction = loan_model.predict(features).tolist()
        proba = loan_model.predict_proba(features).tolist()
        prediction_name = ["Rejected" if p == 0 else "Approved" for p in prediction]

        return {
            "prediction": prediction,
            "prediction_name": prediction_name,
            "proba": proba,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": "PREDICTION_ERROR",
                "message": "Failed to process prediction",
                "details": {"error": str(e)},
            },
        )


@app.post(
    "/loan/predict/batch", response_model=LoanPredictionResponse, tags=["predictions"]
)
async def predict_loan_batch(batch: LoanBatch):
    all_predictions = []
    all_probas = []
    all_prediction_names = []

    for i, loan in enumerate(batch.applications):
        validation_errors = validate_loan_input(loan)
        if validation_errors:
            raise HTTPException(
                status_code=400,
                detail={
                    "error_code": "INVALID_INPUT",
                    "message": f"Validation failed for application at index {i}",
                    "details": {"index": i, "errors": validation_errors},
                },
            )

        try:
            features = preprocess_loan_input(loan)
            prediction = loan_model.predict(features).tolist()
            proba = loan_model.predict_proba(features).tolist()
            prediction_name = ["Rejected" if p == 0 else "Approved" for p in prediction]

            all_predictions.extend(prediction)
            all_probas.extend(proba)
            all_prediction_names.extend(prediction_name)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail={
                    "error_code": "PREDICTION_ERROR",
                    "message": f"Failed to process prediction for application at index {i}",
                    "details": {"index": i, "error": str(e)},
                },
            )

    return {
        "prediction": all_predictions,
        "prediction_name": all_prediction_names,
        "proba": all_probas,
    }
