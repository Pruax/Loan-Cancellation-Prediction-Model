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

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import collections
import time

from sklearn.metrics import classification_report, precision_recall_curve, confusion_matrix, auc, precision_recall_fscore_support
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, average_precision_score
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.utils.fixes import signature
from sklearn.externals import joblib
from sklearn.manifold import TSNE
from sklearn.svm import SVC

from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.metrics import classification_report_imbalanced
from imblearn.under_sampling import NearMiss
from imblearn.datasets import fetch_datasets
from imblearn.over_sampling import SMOTE

from collections import Counter

import warnings
warnings.filterwarnings("ignore")

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

print(df1.head())
print(len(df1))
print(df1['Cancelled'].value_counts())


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

sns.countplot('Cancelled', data=df1, palette=colors)
plt.title('Class Distributions \n (0: Did Not Cancel || 1: Cancelled)', fontsize=14)
plt.show()


# --------------------------------------------------
# Descriptive Statistics
# --------------------------------------------------
# Generate summary statistics for the dataset.

print(df1.describe())


# --------------------------------------------------
# Data Type Inspection
# --------------------------------------------------
# Review column data types to identify categorical
# and numerical variables.

print(df1.dtypes)


# --------------------------------------------------
# Missing Value Analysis
# --------------------------------------------------
# Identify columns containing null or missing values.

print(df1.isnull().sum())


# --------------------------------------------------
# Feature Selection
# --------------------------------------------------
# Select the most relevant variables for predictive
# modeling and analysis.

df = df1[[
    'Cancelled',
    'Accepted_Date',
    'Borrower_Classification',
    'APR',
    'Term',
    'Payments_Rcvd',
    'Borrower_RegisteredOnWeb',
    'Borrower_RegisteredForEForms',
    'Borrower_RegisteredForCancellationWarning'
]]
print(df.head())

# --------------------------------------------------
# Exploratory Data Analysis (EDA)
# --------------------------------------------------
# Analyze feature relationships and identify trends
# associated with loan cancellation behavior.

print(df.head())
print(df.info())
print(df.describe())

# Target variable distribution
print(df["Cancelled"].value_counts())
print(df["Cancelled"].value_counts(normalize=True) * 100)

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

print(correlation_matrix["Cancelled"].sort_values(ascending=False))

# --------------------------------------------------
# Data Preprocessing
# --------------------------------------------------
# Prepare the dataset for machine learning by handling
# categorical variables, scaling, and train/test split.

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Convert categorical variables into numeric columns
model_df = pd.get_dummies(df, drop_first=True)

# Separate features and target
X = model_df.drop("Cancelled", axis=1)
y = model_df["Cancelled"]

# --------------------------------------------------
# Train/Test Split
# --------------------------------------------------
# Split the dataset into training and testing sets.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# Random Forest Model Training
# --------------------------------------------------
# Train a Random Forest classifier to predict
# loan cancellation outcomes.

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

# --------------------------------------------------
# Model Evaluation
# --------------------------------------------------
# Evaluate model performance using recall,
# confusion matrix, and classification metrics.

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Model Performance")
print("-----------------")
print(f"Accuracy:  {accuracy:.2%}")
print(f"Recall:    {recall:.2%}")
print(f"Precision: {precision:.2%}")
print(f"F1 Score:  {f1:.2%}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# --------------------------------------------------
# Confusion Matrix Visualization
# --------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

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
The model achieved approximately 95% recall,
successfully identifying the majority of loans
likely to cancel before maturity.
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