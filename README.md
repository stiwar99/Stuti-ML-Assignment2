# Stuti's Mushroom Edibility Predictor  
### Machine Learning Assignment – 2

**Author:** Stuti Tiwari  
**Registration Number:** 2025AA05728  

---

## 1. Introduction

This project focuses on building a supervised machine learning system capable of determining whether a mushroom is edible or poisonous based on its observable characteristics. The work emphasizes the practical application of classification techniques, from raw data handling to model evaluation and deployment.
Rather than limiting the analysis to a single algorithm, multiple classifiers were implemented and compared to understand how different learning strategies behave when applied to categorical biological data. The inclusion of an interactive interface further demonstrates how machine learning models can be integrated into usable decision-support tools.

---

## 2. Academic Motivation

This project was undertaken as part of Machine Learning Assignment 2 and aims to strengthen practical competence in:

End-to-end machine learning pipeline construction  
Comparative evaluation of classification algorithms  
Metric-driven performance assessment  
Interactive application development  
Cloud-based deployment practices  

The assignment emphasizes reproducibility, transparency, and methodological discipline.

---

## 3. Dataset Specification

### 3.1 Dataset Source

 **Name:** Mushroom Classification Dataset  
 **Repository:** Kaggle (UCI Machine Learning Repository)  
 **Link:** https://www.kaggle.com/datasets/uciml/mushroom-classification  

### 3.2 Dataset Description

The dataset consists of mushroom samples characterized by discrete categorical attributes describing physical structure, odor properties, and habitat conditions. Each instance is annotated with a binary toxicity label.

### 3.3 Statistical Properties

| Attribute | Value |
|-----------|--------|
| Total Samples | 8,124 |
| Input Features | 22 |
| Target Variable | class |
| Classes | Edible (e), Poisonous (p) |
| Missing Values | None |
| Data Type | Categorical |

The dataset exhibits high consistency and structural completeness, making it suitable for supervised learning.

---

## 4. Data Processing Framework

A structured preprocessing pipeline was implemented to ensure numerical compatibility and modeling stability.

### 4.1 Feature Encoding

All categorical attributes were transformed into numerical representations using Label Encoding.One-Hot Encoding was not adopted because it would significantly increase feature dimensionality due to the high cardinality of categorical attributes. Moreover, tree-based classifiers such as Decision Trees, Random Forest, and XGBoost do not require one-hot encoded inputs and can efficiently operate on label-encoded features. Although One-Hot Encoding may benefit linear and distance-based models, the inherent separability of the Mushroom dataset and the dominance of ensemble methods ensured that Label Encoding did not negatively impact performance. Therefore, this design choice represents a trade-off between computational efficiency and representational granularity.

### 4.2 Dataset Partitioning

A stratified train-test split was employed:

Training Set: 80%  
Testing Set: 20%  

Stratification preserved the original class distribution.

### 4.3 Feature Standardization

StandardScaler was applied to normalize feature distributions, enhancing convergence behavior and reducing scale-related bias.

---

## 5. Machine Learning Models

Six supervised classification algorithms were implemented:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbours  
4. Gaussian Naive Bayes  
5. Random Forest Ensemble  
6. XGBoost Ensemble  

All models were trained under identical preprocessing conditions to ensure fair experimental comparison.

---

## 6. Evaluation Methodology

Model performance was assessed using multiple complementary metrics:

Accuracy  
Area Under ROC Curve (AUC)  
Precision  
Recall  
F1 Score  
Matthews Correlation Coefficient (MCC)  

This multidimensional framework enables balanced assessment of predictive reliability and robustness.

---

## 7. Experimental Results

### 7.1 Performance Summary

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|-------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | 0.9625 | 0.9887 | 0.9616 | 0.9604 | 0.9610 | 0.9248 |
| Decision Tree | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| KNN | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Naive Bayes | 0.9286 | 0.9506 | 0.9195 | 0.9336 | 0.9265 | 0.8572 |
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| XGBoost | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

---

## 8. Analytical Observations

### 8.1 Observations about Model Performance

## 8. Analytical Observations

| ML Model Name | Observation on Model Performance |
|---------------|----------------------------------|
| Logistic Regression | Logistic Regression achieved high accuracy and AUC, indicating strong baseline performance. However, its linear decision boundary limits its ability to capture complex non-linear feature interactions present in the dataset. Despite this limitation, the model provides stable and interpretable predictions. |
| Decision Tree | The Decision Tree classifier achieved perfect classification results, demonstrating excellent feature separability. However, the model remains susceptible to overfitting due to hierarchical splitting and may exhibit reduced generalization on unseen datasets. |
| K-Nearest Neighbour | KNN exhibited exceptional predictive accuracy by effectively modeling local neighborhood patterns. Nevertheless, its computational complexity increases significantly with dataset size, making it less scalable for large deployments. |
| Naive Bayes | Naive Bayes delivered comparatively lower performance due to violations of the conditional independence assumption among categorical features. Despite this, it serves as a fast and efficient baseline classifier. |
| Random Forest (Ensemble) | Random Forest achieved optimal performance through ensemble averaging, reducing variance and enhancing robustness. The model effectively captured complex feature relationships while maintaining strong generalization capability. |
| XGBoost (Ensemble) | XGBoost demonstrated outstanding predictive accuracy by iteratively minimizing classification errors and modeling higher-order feature interactions. Its gradient boosting framework enabled superior learning efficiency and stability. |

### 8.2 Summary based on Results

Ensemble-based classifiers demonstrate superior generalization through variance reduction and feature aggregation.  
Decision Tree and KNN models reveal strong separability within the feature space.  
Logistic Regression provides reliable baseline performance but remains limited by linear assumptions.  
Naive Bayes exhibits reduced accuracy due to independence assumption violations.  

The inherent separability of the dataset contributes significantly to high classification performance.

---

## 9. Project Organization

```
Stuti-ML-Assignment2/
└── assests/
   |--mushroom.png               # Mushroom Logo
├── model/                       # Directory for saved models
│   ├── *.pkl                    # Trained model files
├── app.py                       # Streamlit application
├── model_metrics_analysis.csv   # Metrics
├── mushrooms.csv                # Dataset
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── test.csv                     # Sample test Dataset for download
├── train_models.py              # Train all models

```

## 10. Installation and Setup

### 10.1 Prerequisites
Python 3.8 or higher
pip package manager

### 10.2 SetUp Procedure

1. **Clone the repository**
git clone https://github.com/stiwar99/Stuti-ML-Assignment2.git
cd Stuti-ML-Assignment2

2. **Install dependencies**
pip install -r requirements.txt

3. **Train the models**
python train_pipeline.py

   This will:
   Load and preprocess the dataset
   Train all 6 models
   Calculate evaluation metrics
   Save models and metrics to the `model/` directory

4. **Launch Application**
streamlit run app.py

## 11. Usage

### 11.1 Streamlit Application

The Streamlit interface enables users to:
Select classification models
Upload test datasets
Visualize evaluation metrics
Inspect confusion matrices
Review classification reports
Download analytical outputs
The interface is designed to facilitate intuitive exploration of predictive behavior.

## 12. Deployment
The application is deployed using Streamlit Community Cloud, enabling scalable public access and automated redeployment upon repository updates.

### 12.1 Streamlit Community Cloud

**Deployment Steps:**

1. **Push your code to GitHub** (Done)
   git push origin main

2. **Go to Streamlit Cloud**
    Visit: https://streamlit.io/cloud
    Sign in with your GitHub account

3. **Create New App**
   Click "New App" button
   Select repository: `stiwar99/Stuti-ML-Assignment2`
   Choose branch: `main`
   Main file path: `app.py`
   Click "Deploy"

4. **Wait for Deployment**
   Deployment takes 2-5 minutes
   Monitor the deployment logs
   Your application live at: ``


## 13. Evaluation Metrics

Model performance is assessed through a comprehensive set of evaluation criteria. Overall predictive correctness is quantified using accuracy, while discriminatory capability across classes is measured via the area under the receiver operating characteristic curve, implemented in a one-vs-rest configuration for multi-class scenarios. Precision captures the proportion of correctly identified positive predictions relative to all predicted positives, whereas recall reflects the proportion of correctly identified positives among all true positives. The F1 score provides a harmonic balance between precision and recall. We prioritized the F1 score to ensure a balance between missing a poisonous mushroom and wrongly labeling an edible one and the Matthews Correlation Coefficient offers a robust, class-balanced performance indicator suitable for multi-class classification tasks.

## 14. Technical Stack
Python
Streamlit
Scikit-learn
XGBoost
Pandas
NumPy
Matplotlib
Seaborn
Joblib

## 15. Author Information
Stuti Tiwari
Registration Number: 2025AA05728
BITS Pilani
Machine Learning Assignment – 2

## 16. Policy
This repository is intended exclusively for academic evaluation and instructional purposes for ML Assignment 2. Unauthorized commercial utilization is prohibited.

## 17. Acknowledgments
UCI Machine Learning Repository
Kaggle Datasets
BITS Pilani Faculty
Course Instruction Team