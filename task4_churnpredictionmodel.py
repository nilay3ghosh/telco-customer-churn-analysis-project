# -*- coding: utf-8 -*-
"""
Created on Sat Dec 20 20:59:22 2025
Model Training and Comparison for Customer Churn Prediction
@author: USER
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# ---------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------
df = pd.read_csv(
    r"D:\MACHINE_LEARNING___PYTHON\SAIKET_SYSTEM\Telco_Customer_Churn_Dataset  (3).csv"
)

# Convert target column to numeric
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# ---------------------------------------------------------------------
# PREPROCESSING: Handle categorical columns
# ---------------------------------------------------------------------
cat_cols = df.select_dtypes(include=["object"]).columns

label_encoder = LabelEncoder()
for col in cat_cols:
    df[col] = label_encoder.fit_transform(df[col])

# ---------------------------------------------------------------------
# TRAIN–TEST SPLIT
# ---------------------------------------------------------------------
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------------------
# SCALING (Important for Logistic Regression)
# ---------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------
# MODEL DEFINITIONS
# ---------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

# ---------------------------------------------------------------------
# TRAINING AND EVALUATION
# ---------------------------------------------------------------------
for model_name, model in models.items():

    # Logistic Regression → use scaled data
    if model_name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

    print("\n==============================")
    print("MODEL:", model_name)
    print("==============================")
    print("Accuracy:", accuracy_score(y_test, preds))
    print("\nClassification Report:\n", classification_report(y_test, preds))
