# ==================================================
# Loan Cancellation Prediction Model
# Author: Brian Kassin
# ==================================================

# --------------------------------------------------
# Project Overview
# --------------------------------------------------
'''
This project analyzes historical loan data to predict
whether a loan will cancel before reaching maturity.
A Random Forest machine learning model was developed
to identify high-risk loans and uncover the variables
most strongly associated with loan cancellation.
'''

# --------------------------------------------------
# Import Libraries
# --------------------------------------------------

import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE

# --------------------------------------------------
# Import Dataset
# --------------------------------------------------
# Load the loan dataset into a Pandas DataFrame.

df1 = pd.read_csv('Data.csv')

# --------------------------------------------------
# Initial Dataset Inspection
# --------------------------------------------------
# Display the first few rows of the dataset,
# total number of records, and target class counts.

df1.head()
len(df1)
df1['Cancelled'].value_counts()

# --------------------------------------------------
# Analyze Class Distribution
# --------------------------------------------------
# Determine whether the target classes are balanced.
# The dataset contains significantly more non-cancelled
# loans than cancelled loans.

print('Did Not Cancel', round(df1['Cancelled'].value_counts()[0]/len(df1) * 100, 2), '% of the dataset')
print('Cancelled', round(df1['Cancelled'].value_counts()[1]/len(df1) * 100, 2), '% of the dataset')

# --------------------------------------------------
# Visualize Class Distribution
# --------------------------------------------------
# Create a count plot to visualize the imbalance
# between cancelled and non-cancelled loans.

colors = ["#0101DF", "#DF0101"]

sns.countplot(
    x='Cancelled',
    hue='Cancelled',
    data=df1,
    palette=colors,
    legend=False
)

plt.title(
    'Class Distributions\n(0: Did Not Cancel || 1: Cancelled)',
    fontsize=14
)

plt.show()

# --------------------------------------------------
# Descriptive Statistics
# --------------------------------------------------
# Generate summary statistics for the dataset.

df1.describe()

# --------------------------------------------------
# Data Type Inspection
# --------------------------------------------------
# Review column data types to identify categorical
# and numerical variables.

df1.dtypes

# --------------------------------------------------
# Missing Value Analysis
# --------------------------------------------------
# Identify columns containing null or missing values.

df1.isnull().sum()

# --------------------------------------------------
# Feature Selection
# --------------------------------------------------
# Select the most relevant variables for predictive
# modeling and analysis.

df = df1[
    [
        "Cancelled",
        "APR",
        "Term",
        "Premium",
        "Down",
        "AmtFin",
        "FinChg",
        "Payments_Rcvd",
        "Borrower_Classification",
        "Borrower_State",
        "Borrower_RegisteredForCancellationWarning",
        "Borrower_CreditScore",
        "RecurringACH_TF",
    ]
].copy()

df["Borrower_CreditScore"] = pd.to_numeric(
    df["Borrower_CreditScore"],
    errors="coerce"
)

df["Borrower_CreditScore"] = df["Borrower_CreditScore"].fillna(
    df["Borrower_CreditScore"].median()
)

df.head()

# --------------------------------------------------
# Exploratory Data Analysis (EDA)
# --------------------------------------------------
# Analyze feature relationships and identify trends
# associated with loan cancellation behavior.

df.head()
df.info()
df.describe()

# Target variable distribution
df["Cancelled"].value_counts()
df["Cancelled"].value_counts(normalize=True) * 100

sns.countplot(x="Cancelled", data=df)
plt.title("Loan Cancellation Distribution")
plt.show()

# Compare APR by cancellation status
sns.boxplot(x="Cancelled", y="APR", data=df)
plt.title("APR by Loan Cancellation Status")
plt.show()

# Compare loan term by cancellation status
sns.boxplot(x="Cancelled", y="Term", data=df)
plt.title("Loan Term by Cancellation Status")
plt.show()

# Compare payments received by cancellation status
sns.boxplot(x="Cancelled", y="Payments_Rcvd", data=df)
plt.title("Payments Received by Cancellation Status")
plt.show()

# --------------------------------------------------
# Correlation Analysis
# --------------------------------------------------
# Evaluate relationships between numerical variables
# and the target variable using correlation metrics.

numeric_df = df.select_dtypes(include=["int64", "float64"])

correlation_matrix = numeric_df.corr()

plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

print(
    correlation_matrix["Cancelled"]
    .sort_values(ascending=False)
)

# --------------------------------------------------
# Data Preprocessing
# --------------------------------------------------
# Prepare the dataset for machine learning by handling
# categorical variables, scaling, and train/test split.
# Convert categorical variables into numeric columns

model_df = pd.get_dummies(df, drop_first=True)

# Separate features and target
X = model_df.drop("Cancelled", axis=1)
y = model_df["Cancelled"]

# --------------------------------------------------
# Train/Test Split
# --------------------------------------------------
# Split the dataset into training and testing sets.
# Fixed random state for reproducible results.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# Apply SMOTE to Training Data
# --------------------------------------------------
# SMOTE balances the training data by creating synthetic
# examples of the minority class.

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("Before SMOTE:")
print(y_train.value_counts())

print("\nAfter SMOTE:")
print(y_train_smote.value_counts())

# --------------------------------------------------
# Random Forest Model Training
# --------------------------------------------------
# Train a Random Forest classifier to predict
# loan cancellation outcomes.
#
# random_state=42 ensures results are reproducible
# each time the model is trained.

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

rf_model.fit(X_train_smote, y_train_smote)

# Standard predictions
y_pred = rf_model.predict(X_test)

# Probability predictions
y_prob = rf_model.predict_proba(X_test)[:, 1]

# Classification threshold tuned to maximize recall
# while maintaining acceptable precision.
threshold = 0.15

# Adjusted predictions
y_pred_adjusted = (y_prob >= threshold).astype(int)

# --------------------------------------------------
# Model Evaluation
# --------------------------------------------------
# Evaluate model performance using recall,
# confusion matrix, and classification metrics.

accuracy = accuracy_score(y_test, y_pred_adjusted)
recall = recall_score(y_test, y_pred_adjusted)
precision = precision_score(y_test, y_pred_adjusted)
f1 = f1_score(y_test, y_pred_adjusted)

print("Model Performance")
print("-----------------")
print(f"Accuracy:  {accuracy:.2%}")
print(f"Recall:    {recall:.2%}")
print(f"Precision: {precision:.2%}")
print(f"F1 Score:  {f1:.2%}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_adjusted))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_adjusted))

# --------------------------------------------------
# Confusion Matrix Visualization
# --------------------------------------------------

cm = confusion_matrix(y_test, y_pred_adjusted)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# --------------------------------------------------
# Feature Importance Analysis
# --------------------------------------------------
# Identify which variables contribute most strongly
# to predicting loan cancellations.

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print(feature_importance.head(10))

plt.figure(figsize=(10, 6))
sns.barplot(
    x="Importance",
    y="Feature",
    data=feature_importance.head(10)
)
plt.title("Top 10 Most Important Features")
plt.show()

# --------------------------------------------------
# Results & Findings
# --------------------------------------------------
'''
Final Model Performance:

Accuracy: 88.6%
Recall: 97.3%
Precision: 44.0%
F1 Score: 60.6%

The classification threshold was tuned to 15%,
allowing the model to identify nearly all loan
cancellations while maintaining strong overall accuracy.
'''

# --------------------------------------------------
# Conclusion
# --------------------------------------------------
'''
This project demonstrates how machine learning
and data analysis can be leveraged to better
understand customer loan cancellation behavior
and support data-driven business decisions.
```