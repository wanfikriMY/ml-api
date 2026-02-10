import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
import joblib
import os

os.makedirs("./models/iris-model", exist_ok=True)

try:
    df = pd.read_csv("models/iris-model/dataset/Iris.csv")
except FileNotFoundError:
    print(
        "Dataset not found at dataset/Iris.csv, creating dummy data for verification."
    )
    exit(1)

Q1 = df[df.columns[1:5]].quantile(0.25)
Q3 = df[df.columns[1:5]].quantile(0.75)
IQR = Q3 - Q1
df_out = df[
    ~(
        (df[df.columns[1:5]] < (Q1 - 1.5 * IQR))
        | (df[df.columns[1:5]] > (Q3 + 1.5 * IQR))
    ).any(axis=1)
]
df = df_out

X = df.drop(["Id", "Species"], axis=1)
y = df["Species"]

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("classifier", DecisionTreeClassifier(random_state=42)),
    ]
)

pipeline.fit(X_train, y_train)

accuracy = pipeline.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

joblib.dump(pipeline, "models/iris-model/dt_model.pkl")
joblib.dump(le, "models/iris-model/label_encoder.pkl")

print("Artifacts generated successfully.")
print(f"Classes: {le.classes_}")
