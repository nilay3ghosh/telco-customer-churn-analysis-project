📘 Telecom Customer Churn Prediction – README
📌 Project Overview

Customer churn is one of the most critical challenges telecom companies face. Churn refers to customers leaving the service provider.
To reduce churn, companies need to understand:

Which customers are likely to leave?

Why are customers churning?

Which features influence churn the most?

This project builds a complete machine learning pipeline to predict customer churn using the Telco Customer Churn dataset.

🎯 Project Goals

✔ Perform exploratory data analysis (EDA)
✔ Identify patterns between customer attributes and churn
✔ Engineer new features such as Tenure Group and Charge Group
✔ Train multiple ML models and compare performance
✔ Evaluate the best model using ROC–AUC
✔ Visualize feature importance
✔ Create a project-ready pipeline for deployment or demonstration

📂 Dataset Details

Dataset Used: Telco Customer Churn Dataset
It contains customer information such as:

Feature	Description
gender	Male/Female
SeniorCitizen	Whether customer is a senior
Partner, Dependents	Family information
Tenure	Time spent in the company
Contract	Monthly/Yearly contracts
PaymentMethod	Billing preferences
MonthlyCharges	Bill amount
TotalCharges	Entire bill till date
Churn	Target variable (Yes/No)
🧹 Data Preprocessing Performed

Converted Churn from Yes/No → 1/0

Encoded all categorical columns using LabelEncoder

Filled or handled missing values

Created new feature groups:

🔸 Tenure Groups
0–1 year, 1–2 years, 2–4 years, 4–6 years

🔸 Monthly Charges Groups
Low, Medium, High

📊 Exploratory Data Analysis (EDA)

The following plots were generated:

Churn distribution

Gender vs Churn

Partner vs Churn

Tenure distribution

Tenure vs Churn (Boxplot)

Contract type vs Churn

Payment method vs Churn

Monthly charge category vs Churn

Key Insights:

Customers with month-to-month contracts churn more

Customers with high monthly charges churn more

Short tenure (0–1 year) customers are more likely to churn

Electronic check payment users churn at higher rates

🤖 Machine Learning Models Used

Three classification models were trained and compared:

Model	Purpose
Logistic Regression	Baseline model
Decision Tree	Non-linear decision making
Random Forest	Best performance + handles mixed data

The dataset was split using:

Train: 80%
Test: 20%

Scaling used only for Logistic Regression

(Decision Tree and Random Forest do not require scaling)

🏆 Best Model: Random Forest Classifier

Random Forest gave the highest accuracy and most stable performance.

Evaluations performed:

✔ Accuracy
✔ Classification report
✔ ROC Curve
✔ AUC Score
✔ Feature Importance Ranking
📈 ROC–AUC Curve

The ROC curve was plotted using:

roc_curve()

auc()

Higher AUC indicates better performance in distinguishing churn vs non-churn customers.

⭐ Top Features Influencing Churn

Using Random Forest’s feature importance:

MonthlyCharges

Contract

Tenure

InternetService

TechSupport

PaymentMethod

OnlineSecurity

These features are critical for customer retention strategies.

🗂 Project Folder Structure
📁 Telco-Churn-Prediction
│── README.md
│── dataset.csv
│── eda_plots/
│── model_training.py
│── churn_analysis.ipynb
│── roc_feature_importance.py
│── requirements.txt

▶️ How to Run the Project
1. Install Required Libraries
pip install pandas numpy seaborn matplotlib scikit-learn

2. Run EDA
python churn_analysis.ipynb

3. Train Models
python model_training.py

4. Generate ROC Curve + Feature Importance
python roc_feature_importance.py

📝 Technologies Used

Python

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn


🏁 Conclusion

This project successfully predicts telecom customer churn using machine learning.
Using Random Forest and proper feature analysis, we can identify the customers who are most likely to leave and take preventive actions.