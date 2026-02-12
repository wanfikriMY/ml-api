#!/usr/bin/env python3
"""
Loan Approval Prediction using RandomForestClassifier
=====================================================
This script trains a RandomForest model to predict loan approval status (Y/N)
based on applicant demographic and financial features.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import joblib
import os
import warnings

warnings.filterwarnings("ignore")

DATASET_PATH = "/Users/wanfikri/Repo/private/ml-api/prediction_model/loan-approval/datasets/LoanApprovalPrediction.csv"
OUTPUT_DIR = "/Users/wanfikri/Repo/private/ml-api/models/loan-approval/results"
MODEL_PATH = os.path.join(OUTPUT_DIR, "random_forest_model.joblib")
PREDICTIONS_PATH = os.path.join(OUTPUT_DIR, "predictions.csv")

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
TARGET_COLUMN = "Loan_Status"


def load_dataset(path):

    df = pd.read_csv(path)
    return df


def explore_data(df):
  
    for col in df.columns:
        missing = df[col].isnull().sum()
        if missing > 0:
            print(f"    - {col}: {missing} ({100 * missing / len(df):.1f}%)")


def preprocess_data(df):
  

    df_processed = df.copy()

    if "Loan_ID" in df_processed.columns:
        df_processed = df_processed.drop("Loan_ID", axis=1)
        print("    Dropped 'Loan_ID' column")

    for col in CATEGORICAL_COLUMNS:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
            print(f"    Imputed '{col}' with mode")

    for col in NUMERICAL_COLUMNS:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors="coerce")
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())
            print(f"    Imputed '{col}' with median")

    print(f"\n    DataFrame dtypes after imputation:")
    for col in df_processed.columns:
        print(f"    - {col}: {df_processed[col].dtype}")

    print(f"\n    First 3 rows after imputation:")
    print(df_processed.head(3).to_string())

    label_encoders = {}
    object_cols = [
        col
        for col in df_processed.columns
        if df_processed[col].dtype in ["object", "str"]
    ]
    print(f"    Object/string columns before encoding: {object_cols}")
    for col in df_processed.columns:
        if df_processed[col].dtype in ["object", "str"] and col != TARGET_COLUMN:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            label_encoders[col] = le
            print(f"    Encoded '{col}': {list(le.classes_)}")

    df_processed[TARGET_COLUMN] = df_processed[TARGET_COLUMN].map({"Y": 1, "N": 0})

    print(f"\n    Preprocessed dataset shape: {df_processed.shape}")
    return df_processed, label_encoders


def train_model(df_processed):
    """Train the RandomForestClassifier model."""
    print("\n[4] Training RandomForestClassifier")
    print("-" * 40)

    X = df_processed.drop(TARGET_COLUMN, axis=1)
    y = df_processed[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"    Training set: {X_train.shape[0]} samples")
    print(f"    Test set: {X_test.shape[0]} samples")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    print("    Training model...")
    model.fit(X_train, y_train)
    print("    Model trained successfully!")

    return model, X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test):
    """Evaluate the model and display results."""
    print("\n[5] Model Evaluation")
    print("-" * 40)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n    Performance Metrics:")
    print(f"    - Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"    - Precision: {precision:.4f}")
    print(f"    - Recall:    {recall:.4f}")
    print(f"    - F1-Score:  {f1:.4f}")

    cm = confusion_matrix(y_test, y_pred)

    print(f"\n    Classification Report:")
    print(
        classification_report(
            y_test, y_pred, target_names=["Rejected (N)", "Approved (Y)"]
        )
    )

    return y_pred, cm


def display_feature_importance(model, X):
    """Display feature importance rankings."""
    print("\n[6] Feature Importance")
    print("-" * 40)

    feature_importance = pd.DataFrame(
        {"Feature": X.columns, "Importance": model.feature_importances_}
    ).sort_values("Importance", ascending=False)

    print(f"\n    {'Rank':<6}{'Feature':<20}{'Importance':<12}")
    print(f"    {'-' * 38}")
    for idx, row in enumerate(feature_importance.itertuples(), 1):
        print(f"    {idx:<6}{row.Feature:<20}{row.Importance:.4f}")

    return feature_importance


def save_results(model, feature_importance, X_test, y_test, y_pred, label_encoders):

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    print(f"    Model saved to: {MODEL_PATH}")

    predictions_df = pd.DataFrame(
        {
            "Actual": y_test.values,
            "Predicted": y_pred,
            "Correct": y_test.values == y_pred,
        }
    )
    predictions_df.to_csv(PREDICTIONS_PATH, index=False)
    print(f"    Predictions saved to: {PREDICTIONS_PATH}")

    feature_importance.to_csv(
        os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False
    )

    joblib.dump(label_encoders, os.path.join(OUTPUT_DIR, "label_encoders.joblib"))


def main():
    df = load_dataset(DATASET_PATH)
    df_processed, label_encoders = preprocess_data(df)
    model, X_train, X_test, y_train, y_test = train_model(df_processed)
    y_pred, cm = evaluate_model(model, X_test, y_test)
    feature_importance = display_feature_importance(model, X_train)
    save_results(model, feature_importance, X_test, y_test, y_pred, label_encoders)



if __name__ == "__main__":
    main()
