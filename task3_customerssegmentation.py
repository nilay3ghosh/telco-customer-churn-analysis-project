
# -*- coding: utf-8 -*-
"""
Customer Churn Segmentation Analysis
Author : USER
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv(
    r"D:\MACHINE_LEARNING___PYTHON\SAIKET_SYSTEM\Telco_Customer_Churn_Dataset  (3).csv"
)

# Convert churn from Yes/No to 1/0 for calculations
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# ---- Tenure-Based Segmentation ----
df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[0, 12, 24, 48, 72],
    labels=["0–1 year", "1–2 years", "2–4 years", "4–6 years"]
)

# ---- Monthly Charge Segmentation ----
df["ChargeGroup"] = pd.cut(
    df["MonthlyCharges"],
    bins=[0, 35, 70, 100],
    labels=["Low", "Medium", "High"]
)

# ---- Average Churn % by Tenure Segment ----
tenure_churn = df.groupby("TenureGroup")["Churn"].mean() * 100
print("\nChurn Rate by Tenure Group (%)")
print(tenure_churn)

# ---- Average Churn % by Monthly Charges ----
charges_churn = df.groupby("ChargeGroup")["Churn"].mean() * 100
print("\nChurn Rate by Monthly Charge Group (%)")
print(charges_churn)

# ---- Churn Rate by Contract Type ----
contract_churn = df.groupby("Contract")["Churn"].mean() * 100
print("\nChurn Rate by Contract Type (%)")
print(contract_churn)

# ---- Plot: Monthly Charge Groups vs Churn ----
sns.countplot(data=df, x="ChargeGroup", hue="Churn")
plt.title("Customer Churn by Monthly Charge Category")
plt.xlabel("Monthly Charge Group")
plt.ylabel("Count")
plt.show()
