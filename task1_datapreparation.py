import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("D:\MACHINE_LEARNING___PYTHON\SAIKET_SYSTEM\Telco_Customer_Churn_Dataset  (3).csv")

# ---- Basic Info ----
print(df.head())
print(df.info())
print(df.describe())

# ---- Handling Missing Values ----
df.replace(" ", np.nan, inplace=True)
df.fillna(df.median(numeric_only=True), inplace=True)

# ---- Encoding Categorical Variables ----
encoder = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = encoder.fit_transform(df[col])

# ---- Train-Test Split ----
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train Size:", X_train.shape)
print("Test Size:", X_test.shape)
