# Bank Customer Churn Prediction with Deep Learning

A machine learning project focused on predicting customer churn using **Artificial Neural Networks** in TensorFlow-Keras. This notebook explores performance optimization, handles class imbalance (using SMOTE, over- & under-sampling), and evaluates various architectures using metrics like accuracy, F1-score, and confusion matrix.


## 📂 Dataset

- **Source**: `bank_churn.csv`  
- **Target**: `Exited` (1 if the customer churned, 0 otherwise)


## 🔍 Problem Statement

To build a robust machine learning model that can accurately predict if a customer is likely to churn, helping the bank take preventive actions.

## ⚙️ Workflow Overview

### 🔹 Data Preprocessing
- Removed identifiers (`CustomerID`, `RowNumber`, etc.)
- Applied One-Hot Encoding for categorical variables
- Scaled continuous features with `MinMaxScaler`

### 🔹 Model Building & Tuning
- Built 4 different Keras Sequential models:
  - **Base Model**
  - **Dropout Regularization**
  - **Batch Normalization**
  - **Training Speed Comparison (CPU vs GPU)**

### 🔹 Evaluation
- Accuracy, Confusion Matrix, Precision, Recall, and F1-Score

### 🔹 Class Imbalance Handling
- ✅ **Undersampling**
- ✅ **Oversampling**
- ✅ **SMOTE (Synthetic Minority Oversampling)**  
- Compared how each technique improves classification performance on the minority class.

---

## 🧪 Model Architectures

| Model Variation             | Layers Used                 | Notes                              |
|----------------------------|-----------------------------|-------------------------------------|
| Base Model                 | Dense, ReLU + Sigmoid       | Baseline performance                |
| With Dropout               | Dropout layers              | Regularization to prevent overfitting |
| With Batch Normalization   | BatchNormalization layers   | Stabilize and accelerate training  |
| With SMOTE                 | Same model as base          | Improved recall on minority class  |

---

## 📊 Results Snapshot

| Approach           | Accuracy | Recall | F1-Score |
|-------------------|-----------|--------|----------|
| Base Model        | ~84%     |  Low    |    Low   |
| Dropout           | ~82%     |  Low    |    Low   |
| Batch Norm        | ~84%     |  Low    |    Low   |
| Undersampling     | ~71%     |  Good   |    ~70%  |
| Oversampling      | ~74%     |  Better |    ~73%  |
| SMOTE             | ~74%     |  Better |    ~75%  |

---

## 🛠️ Tech Stack

- **Languages**: Python  
- **Frameworks**: TensorFlow, Keras  
- **Libraries**: NumPy, Pandas, Scikit-learn, imbalanced-learn, Matplotlib  

---

## 🚀 How to Run

1. Clone the repo:
   ```
   git clone https://github.com/vaibhavr54/bank-churn-prediction-deep-learning.git
   cd bank-churn-prediction-deep-learning

2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Open the Jupyter Notebook:

   ```
   jupyter notebook
   ```

---

## 📬 Contact

Feel free to connect with me on [LinkedIn](https://linkedin.com/in/vaibhav-rakshe-7309aa2a5)
or drop a ⭐ on [GitHub](https://github.com/vaibhavr54)!

---
