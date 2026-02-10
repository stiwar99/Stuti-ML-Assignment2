"""
STUTI TIWARI -- 2025AA05728
ML Assignment 2
Mushroom Edibility Prediction (Binary Classification)

Description:
This project evaluates multiple machine learning classifiers to predict
whether a mushroom is edible or poisonous using categorical attributes.
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Utilities
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score,
    f1_score, matthews_corrcoef
)

# Inbuilt Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# DATA LOADING

def load_dataset(file_path):
    """Load dataset and display info."""
    dataset = pd.read_csv(file_path)
    print("Dataset loaded successfully")
    print("Shape:", dataset.shape)
    print(dataset.head())
    return dataset


#  DATA PREPROCESSING

def encode_features(dataset):
    """
    Encode categorical variables using Label Encoding.
    Since all features are categorical, numerical conversion is required.
    """
    encoder = LabelEncoder()
    encoded_dataset = dataset.copy()

    for column in encoded_dataset.columns:
        encoded_dataset[column] = encoder.fit_transform(encoded_dataset[column])

    print("Categorical features encoded")
    return encoded_dataset


# MODEL EVALUATION

def evaluate_model(true_labels, predictions, probabilities):
    """Compute multiple evaluation metrics for a trained model."""
    return {
        "Accuracy": accuracy_score(true_labels, predictions),
        "AUC": roc_auc_score(true_labels, probabilities),
        "Precision": precision_score(true_labels, predictions),
        "Recall": recall_score(true_labels, predictions),
        "F1_Score": f1_score(true_labels, predictions),
        "MCC": matthews_corrcoef(true_labels, predictions)
    }


# STUTI's TRAIN PIPELINE

print("\n========== MUSHROOM CLASSIFICATION PIPELINE ==========\n")

# Dataset: Kaggle Mushroom Classification Dataset
# Source: https://www.kaggle.com/datasets/uciml/mushroom-classification

data = load_dataset("mushrooms.csv")

# Check for missing values
print("\nMissing Value Summary:")
print(data.isnull().sum())

# Encode data
data_encoded = encode_features(data)

# Feature-target separation
features = data_encoded.drop("class", axis=1)
target = data_encoded["class"]

# Train-test split with stratification to preserve class balance
X_train, X_test, Y_train, Y_test = train_test_split(
    features,
    target,
    test_size=0.20,
    random_state=42,
    stratify=target
)

# Feature scaling for Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Data preprocessing completed\n")

# MODELS

model_registry = {
    "Logistic_Regression": LogisticRegression(max_iter=2000),
    "Decision_Tree_Classifier": DecisionTreeClassifier(random_state=42),
    "K_Nearest_Neighbour": KNeighborsClassifier(n_neighbors=5),
    "Naive_Bayes": GaussianNB(),
    "Ensemble-Random_Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Ensemble-XGBoost": XGBClassifier(eval_metric="logloss")
}

# Directory for model pkl

os.makedirs("model", exist_ok=True)

performance_summary = []

print("Model training and evaluation initiated...\n")

for model_name, model in model_registry.items():

    print(f"Training model: {model_name}")

    model.fit(X_train_scaled, Y_train)

    predictions = model.predict(X_test_scaled)
    probabilities = model.predict_proba(X_test_scaled)[:, 1]

    metrics = evaluate_model(Y_test, predictions, probabilities)

    performance_summary.append([
        model_name,
        metrics["Accuracy"],
        metrics["AUC"],
        metrics["Precision"],
        metrics["Recall"],
        metrics["F1_Score"],
        metrics["MCC"]
    ])

    joblib.dump(model, f"model/{model_name.replace(' ', '_')}.pkl")
    print(f"{model_name} saved successfully\n")


# ANALYSIS

stuti_metrics = pd.DataFrame(
    performance_summary,
    columns=["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
)

stuti_metrics.to_csv("model_metrics_analysis.csv", index=False)

test_output = pd.DataFrame(X_test_scaled, columns=features.columns)
test_output["class"] = Y_test.values
test_output.to_csv("test.csv", index=False)

print("Evaluation completed successfully")
print(stuti_metrics)