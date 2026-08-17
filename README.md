# Machine Learning Assignment - 2: Classification Modeling & Streamlit Deployment

**Course**: M.Tech (AIML / DSE)  
**Division**: Work Integrated Learning Programmes Division, BITS Pilani  
**Course Name**: Machine Learning  
**Assignment**: Assignment - 2  

---

## a. Problem Statement

Direct marketing campaigns (phone calls) of a banking institution are often inefficient when targeting untargeted customer segments. The objective of this project is to build, evaluate, and compare **5 Machine Learning Classification Models** to predict whether a client will subscribe to a bank term deposit (Target Variable: `y` - Binary Classification: `yes` / `no`). 

Accurately predicting subscription probabilities enables banks to optimize marketing resources, increase campaign conversion rates, and reduce operational costs by contacting high-probability prospects.

---

## b. Dataset Description

The dataset used in this assignment is the benchmark **Bank Marketing Dataset** from the **UCI Machine Learning Repository**.

- **Dataset Source**: UCI Machine Learning Repository / Kaggle
- **Total Instances**: 2,000 samples (Test dataset partition: 400 samples) — *Exceeds minimum requirement of 500 instances*
- **Total Features**: 16 input features + 1 target variable — *Exceeds minimum requirement of 12 features*

### Feature Details:
1. `age`: Age of customer (Numeric)
2. `job`: Type of job (Categorical: admin, blue-collar, technician, etc.)
3. `marital`: Marital status (Categorical: married, single, divorced)
4. `education`: Education level (Categorical: primary, secondary, tertiary, unknown)
5. `default`: Has credit in default? (Categorical: no, yes)
6. `balance`: Average yearly balance in euros (Numeric)
7. `housing`: Has housing loan? (Categorical: no, yes)
8. `loan`: Has personal loan? (Categorical: no, yes)
9. `contact`: Communication contact type (Categorical: cellular, telephone, unknown)
10. `day`: Last contact day of the month (Numeric)
11. `month`: Last contact month of year (Categorical: jan, feb, ..., dec)
12. `duration`: Last contact duration in seconds (Numeric)
13. `campaign`: Number of contacts performed during campaign (Numeric)
14. `pdays`: Days after client was last contacted (-1 means not previously contacted) (Numeric)
15. `previous`: Number of contacts performed before this campaign (Numeric)
16. `poutcome`: Outcome of the previous marketing campaign (Categorical: unknown, failure, other, success)
17. `y` / `Target`: Has the client subscribed to a term deposit? (Binary Target: 1 = Yes, 0 = No)

---

## c. GitHub Repository Link

- **GitHub Repository**: `https://github.com/Asna-Ahamed/bits-ml-assignment-2`
- **Live Streamlit App**: `https://ml-assignment-2.streamlit.app` (Deploy on Streamlit Cloud)

### Repository Structure:
```
ml_assignment_2/
├── app.py                      # Interactive Streamlit Web Application
├── train_and_evaluate.py       # ML Pipeline script (Preprocessing, training, evaluation)
├── requirements.txt            # Python package dependencies
├── README.md                   # Assignment Report & Documentation
├── test_data.csv               # Test dataset CSV for evaluation & app testing
└── model/                      # Serialized trained model binaries
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    ├── scaler.pkl
    └── encoders.pkl
```

---

## d. Models Used & Performance Evaluation

All 5 classification models were trained and evaluated on the same test partition (`test_data.csv`). The 6 evaluation metrics computed for each model are summarized below:

### 1. Model Evaluation Metrics Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.8550** | **0.9309** | 0.8684 | **0.9094** | **0.8885** | **0.6831** |
| **Decision Tree** | 0.7825 | 0.8333 | 0.8249 | 0.8346 | 0.8297 | 0.5288 |
| **kNN** | 0.7450 | 0.8121 | 0.7676 | 0.8583 | 0.8104 | 0.4310 |
| **Naive Bayes** | 0.8400 | 0.9150 | **0.8800** | 0.8661 | 0.8730 | 0.6570 |
| **Random Forest (Ensemble)** | 0.8425 | 0.9297 | 0.8631 | 0.8937 | 0.8781 | 0.6565 |

---

### 2. Model Performance Observations

| ML Model Name | Observation about Model Performance |
| :--- | :--- |
| **Logistic Regression** | Achieved the highest overall performance with an **Accuracy of 85.50%**, **AUC of 0.9309**, **F1 Score of 0.8885**, and **MCC of 0.6831**. Due to feature standardization, it effectively captures the linear decision boundary between financial metrics (balance, duration) and subscription likelihood. |
| **Decision Tree** | Yielded moderate results with **78.25% Accuracy** and **0.8333 AUC**. While highly interpretable, simple decision trees tend to overfit high-cardinality features like `day` and `balance` without regularization. |
| **kNN** | Recorded an **Accuracy of 74.50%** and **MCC of 0.4310**. Performance suffers due to distance metrics becoming less discriminative in 16-dimensional feature space (curse of dimensionality) and sensitivity to local noise. |
| **Naive Bayes** | Demonstrated strong performance with **84.00% Accuracy**, **0.9150 AUC**, and the highest **Precision (88.00%)**. Despite its independence assumption, Gaussian Naive Bayes handled continuous numerical attributes well and trained extremely fast. |
| **Random Forest (Ensemble)** | Showed strong ensemble performance with **84.25% Accuracy**, **0.9297 AUC**, **F1 of 0.8781**, and **MCC of 0.6565**. Combining multiple decision trees effectively mitigated variance and provided balanced precision and recall. |

---

### 3. Overall Winner Selection

🏆 **Overall Winner: Logistic Regression** (followed closely by **Random Forest Ensemble**).

**Justification**:
1. **Highest MCC (0.6831)**: Matthews Correlation Coefficient is the most robust single metric for binary classification as it accounts for true and false positives and negatives. Logistic Regression achieved the highest MCC score.
2. **Superior AUC (0.9309)**: Demonstrates exceptional class separation across all decision thresholds.
3. **Balanced Recall & F1 (0.9094 & 0.8885)**: In bank marketing, identifying true potential subscribers (high recall) while minimizing false promises is critical. Logistic Regression achieved the highest recall and F1 score among all models.

---

## How to Run Locally

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Asna-Ahamed/bits-ml-assignment-2.git
   cd bits-ml-assignment2
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train Models & Generate Test Data**:
   ```bash
   python train_and_evaluate.py
   ```

4. **Launch Streamlit Web App**:
   ```bash
   streamlit run app.py
   ```
