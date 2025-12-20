# -*- coding: utf-8 -*-
"""
Created on Sat Dec 20 20:31:15 2025
@author: USER
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv(r"D:\MACHINE_LEARNING___PYTHON\SAIKET_SYSTEM\Telco_Customer_Churn_Dataset  (3).csv")

# Display basic information
print(df.head())
print(df.columns)

# Convert Churn to numeric (Yes=1, No=0)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# ---- Churn Rate ----
churn_rate = df["Churn"].mean() * 100
print("Overall Churn Rate:", churn_rate)

sns.countplot(data=df, x="Churn")
plt.title("Churn Distribution")
plt.show()

# ---- Gender Distribution ----
sns.countplot(data=df, x="gender", hue="Churn")
plt.title("Gender vs Churn")
plt.show()

# ---- Partner Status ----
sns.countplot(data=df, x="Partner", hue="Churn")
plt.title("Partner Status vs Churn")
plt.show()

# ---- Tenure Distribution ----
sns.histplot(df["tenure"], kde=True)
plt.title("Tenure Distribution")
plt.show()

# ---- Tenure vs Churn ----
sns.boxplot(data=df, x="Churn", y="tenure")
plt.title("Tenure vs Churn")
plt.show()

# ---- Contract Type ----
sns.countplot(data=df, x="Contract", hue="Churn")
plt.title("Contract Type vs Churn")
plt.show()

# ---- Payment Method ----
sns.countplot(data=df, x="PaymentMethod", hue="Churn")
plt.xticks(rotation=45)
plt.title("Payment Method vs Churn")
plt.show()
