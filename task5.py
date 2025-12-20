

# -*- coding: utf-8 -*-
"""
ROC Curve, AUC Score, and Feature Importance for Churn Model
Author: USER
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score

# --------------------------------------------------------------------
# LOAD DATA
# --------------------------------------------------------------------
df = pd.read_csv(
    r"D:\MACHINE_LEARNING___PYTHON\SAIKET_SYSTEM\Telco_Customer_Churn_Dataset  (3).csv"
)

# Convert Churn to numeric
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Encode categorical columns
cat_cols = df.select_dtypes(include=["object"]).columns
encoder = LabelEncoder()
for col in cat_cols:
    df[col] = encoder.fit_transform(df[col])

# --------------------------------------------------------------------
# SPLIT DATA
# --------------------------------------------------------------------
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------------------------------------------
# BEST MODEL: RANDOM FOREST
# --------------------------------------------------------------------
best_model = RandomForestClassifier(n_estimators=200, random_state=42)
best_model.fit(X_train, y_train)

# Predictions for ROC Curve
y_prob = best_model.predict_proba(X_test)[:, 1]

# --------------------------------------------------------------------
# ROC CURVE + AUC
# --------------------------------------------------------------------
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, linewidth=2)
plt.plot([0, 1], [0, 1], 'k--')  # diagonal line
plt.title(f"ROC Curve (AUC = {roc_auc:.4f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.grid(True)
plt.show()

# --------------------------------------------------------------------
# FEATURE IMPORTANCE PLOT
# --------------------------------------------------------------------
importances = pd.Series(best_model.feature_importances_, index=X.columns)

plt.figure(figsize=(9, 5))
importances.nlargest(10).plot(kind="bar")
plt.title("Top 10 Features Influencing Customer Churn")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.show()

# --------------------------------------------------------------------
# PRINT ACCURACY (Optional)
# --------------------------------------------------------------------
y_pred = best_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
