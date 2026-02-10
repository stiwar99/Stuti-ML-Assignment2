# Mushroom Edibility Prediction Streamlit App
# Author: 2025AA05728 (STUTI TIWARI)

import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import streamlit.components.v1 as components

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)
from PIL import Image

# Page

st.set_page_config(
    page_title="Stuti's Mushroom Edibility Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Stuti's Signature Style UI

st.markdown("""
<style>

html, body {
    height: 100%;
    margin: 0;
    padding: 0;
    font-family: "Poppins", sans-serif;
}

body {
    background: linear-gradient(
        135deg,
        #fbc2eb,
        #fad0c4,
        #cdb4db,
        #a18cd1
    );
    background-size: 300% 300%;
    animation: bgMove 20s ease infinite;
}

@keyframes bgMove {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.stApp {
    background: transparent;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        rgba(255,200,220,0.7),
        rgba(200,180,255,0.7)
    );

    backdrop-filter: blur(12px);
    border-right: 1px solid rgba(255,255,255,0.4);
}

section[data-testid="stSidebar"] > div {
    margin-top: 40px;
}

section[data-testid="stSidebar"] * {
    color: #2d033b !important;
}

/* Main container */
.block-container {
    max-width: 1150px;
    padding-top: 2rem;
}

.stFileUploader,
.stDataFrame,
.stAlert,
div[role="radiogroup"] {
    background: rgba(255,255,255,0.75);
    border-radius: 18px;
    padding: 15px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.15);
}

button {
    background: linear-gradient(90deg,#ff758c,#c77dff) !important;
    color: white !important;
    border-radius: 22px !important;
    font-weight: bold !important;
}

.sidebar-mushroom {
    width: 100px;
    margin: 0 auto 15px auto;
    display: block;

    filter: drop-shadow(0 0 8px #ff7ac7);

    animation: glowPulse 2.5s infinite alternate;
}

@keyframes glowPulse {
    from {
        filter: drop-shadow(0 0 6px #ff7ac7);
    }
    to {
        filter: drop-shadow(0 0 20px #c77dff);
    }
}

</style>
""", unsafe_allow_html=True)


# Unique dynamic floating Mushrooms idea as mine project is Mushroom Predictor.

st.markdown(
    """
    <style>

    .mushroom {
        position: fixed;
        font-size: 18px;
        opacity: 0.25;
        animation: floatUp linear infinite;
        pointer-events: none;
        z-index: 0;
    }

    .m1  { left: 6%;  animation-duration: 16s; animation-delay: -2s; top: 95%; }
    .m2  { left: 15%; animation-duration: 18s; animation-delay: -6s; top: 80%; }
    .m3  { left: 25%; animation-duration: 14s; animation-delay: -4s; top: 70%; }
    .m4  { left: 35%; animation-duration: 20s; animation-delay: -8s; top: 90%; }
    .m5  { left: 45%; animation-duration: 17s; animation-delay: -10s; top: 65%; }

    .m6  { left: 55%; animation-duration: 19s; animation-delay: -5s; top: 75%; }
    .m7  { left: 65%; animation-duration: 15s; animation-delay: -12s; top: 95%; }
    .m8  { left: 75%; animation-duration: 18s; animation-delay: -7s; top: 70%; }
    .m9  { left: 85%; animation-duration: 22s; animation-delay: -9s; top: 85%; }
    .m10 { left: 95%; animation-duration: 16s; animation-delay: -3s; top: 65%; }

    @keyframes floatUp {
        from { transform: translateY(0); }
        to   { transform: translateY(-120vh); }
    }

    </style>

    <div class="mushroom m1">🍄</div>
    <div class="mushroom m2">🍄</div>
    <div class="mushroom m3">🍄</div>
    <div class="mushroom m4">🍄</div>
    <div class="mushroom m5">🍄</div>
    <div class="mushroom m6">🍄</div>
    <div class="mushroom m7">🍄</div>
    <div class="mushroom m8">🍄</div>
    <div class="mushroom m9">🍄</div>
    <div class="mushroom m10">🍄</div>
    """,
    unsafe_allow_html=True
)


# Title matching with mushrooms

st.markdown(
    """
    <style>

    .mushroom-title {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        margin: 15px 0 18px 0;
        letter-spacing: 1px;

        /* Soft red → cream gradient */
        background: linear-gradient(
            90deg,
            #ff6b6b,
            #ff9a9a,
            #fff0f0,
            #ff9a9a,
            #ff6b6b
        );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;

        /* Thin soft outline */
        -webkit-text-stroke: 0.5px rgba(0,0,0,0.5);

        /* Subtle glow */
        text-shadow: 0 1px 3px rgba(255,100,100,0.3);
    }

    </style>

    <div class="mushroom-title">
        Stuti's Mushroom Edibility Predictor
    </div>
    """,
    unsafe_allow_html=True
)

components.html(
    """
    <style>

    .center-info {
        max-width: 900px;
        margin: 0 auto 35px auto;
        text-align: center;
        font-family: Poppins, sans-serif;
    }

    .center-info-caption {
        font-size: 18px;
        font-weight: 700;
        color: #6a0572;
        margin-bottom: 8px;
        letter-spacing: 0.5px;
    }

    .center-info-text {
        font-size: 17px;
        font-weight: 500;
        color: #3c096c;
        line-height: 1.7;
    }

    </style>

    <div class="center-info">

        <div class="center-info-caption">
            Machine Learning Assignment — 2
        </div>

        <div class="center-info-text">
            This web application evaluates whether a given mushroom sample is
            Edible or Poisonous using Machine Learning models.
        </div>

    </div>
    """,
    height=120,
)
# ---------------- FORCE DARK TABLES (STREAMLIT SAFE) ----------------

st.markdown("""
<style>

/* DataFrame & Table container */
div[data-testid="stDataFrame"],
div[data-testid="stTable"] {
    background-color: #0f172a !important;
    border-radius: 14px;
    padding: 6px;
}

/* Header cells */
div[data-testid="stDataFrame"] th,
div[data-testid="stTable"] th {
    background-color: #1e293b !important;
    color: #f8fafc !important;
    font-weight: 700 !important;
    border-bottom: 1px solid #334155 !important;
}

/* Body cells */
div[data-testid="stDataFrame"] td,
div[data-testid="stTable"] td {
    background-color: #0f172a !important;
    color: #e5e7eb !important;
    border-bottom: 1px solid #1e293b !important;
    font-weight: 500 !important;
}

/* Row hover */
div[data-testid="stDataFrame"] tr:hover td {
    background-color: #1e293b !important;
}

</style>
""", unsafe_allow_html=True)


# Model Selection Dropdown

img = Image.open("assets/mushroom.png")

st.sidebar.image(
    img,
    use_container_width=True
)

st.sidebar.title("Model Selection Interface")

available_models = [
    "Logistic_Regression",
    "Decision_Tree_Classifier",
    "K_Nearest_Neighbour",
    "Naive_Bayes",
    "Ensemble-Random_Forest",
    "Ensemble-XGBoost"
]

chosen_model = st.sidebar.radio(
    "Hand-pick the model of your Preference:",
    available_models
)


# Sample Dataset Download button

url = "https://raw.githubusercontent.com/stiwar99/Stuti-ML-Assignment2/main/test.csv"
response = requests.get(url)

st.sidebar.download_button(
    label="Download test.csv",
    data=response.content,
    file_name="test.csv",
    mime="text/csv"
)


# Fetching Trained Models

@st.cache_resource(show_spinner=False)
def get_trained_model(model_name: str):

    path = os.path.join("model", f"{model_name}.pkl")

    if not os.path.isfile(path):
        return None

    return joblib.load(path)


classifier = get_trained_model(chosen_model)

if classifier is None:
    st.error(f"Trained model file not found for: {chosen_model}")
    st.stop()


# Dataset Upload Option

st.subheader("Upload Sample Dataset")

uploaded_file = st.file_uploader(
    "Upload CSV file for evaluation",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload a valid test.csv file to continue.")
    st.stop()


df = pd.read_csv(uploaded_file)

st.markdown("### Preview of Uploaded Dataset")
st.dataframe(df.head())


# Expected Dataset

if "class" not in df.columns:
    st.error("Target column 'class' not found in dataset.")
    st.stop()

X = df.drop(columns=["class"])
y = df["class"]


# Results on Test Data

y_pred = classifier.predict(X)

try:
    y_prob = classifier.predict_proba(X)[:, 1]
except:
    y_prob = None


# Model Metrics Analysis

st.subheader("Model Performance Analysis")

accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)

auc = roc_auc_score(y, y_prob) if y_prob is not None else np.nan
mcc = matthews_corrcoef(y, y_pred)

# ---------------- FIX TABLE VISIBILITY ----------------

metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "AUC", "Matthews Correlation Coefficient (MCC Score)"],
    "Score": [accuracy, precision, recall, f1, auc, mcc]
})


st.table(metrics_df.style.format({"Score": "{:.4f}"}))


# Display Confusion Matrix

st.subheader("Confusion Matrix")

cm = confusion_matrix(y, y_pred)
fig, ax = plt.subplots(figsize=(2.8,2.8), dpi=130)

sns.heatmap(
    cm,
    annot=True,
    cmap="coolwarm",
    fmt="d",
    cbar=False,
    ax=ax,
    annot_kws={"size":9}
)

ax.set_xlabel("Predicted", fontsize=9)
ax.set_ylabel("Actual", fontsize=9)

ax.tick_params(axis='both', labelsize=8)

plt.tight_layout()

st.pyplot(fig, use_container_width=False)


# Results Summary

st.subheader("Classification Summary")

report = classification_report(
    y,
    y_pred,
    output_dict=True
)

report_df = pd.DataFrame(report).T

st.dataframe(report_df.round(4))