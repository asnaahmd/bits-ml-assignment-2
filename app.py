import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_curve
)

# Set page config
st.set_page_config(
    page_title="Bank Term Deposit Classifier - ML Assignment 2",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern design
st.markdown("""
<style>
    /* Main layout & card styles */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748B;
        text-align: center;
        margin-bottom: 1.8rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #2563EB;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        white-space: pre-wrap;
        border-radius: 8px;
        padding: 0 16px;
    }
</style>
""", unsafe_allow_html=True)

# App Title & Header
st.markdown("<div class='main-header'>🏦 Bank Deposit Classification & ML Evaluation Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>M.Tech (AIML/DSE) Machine Learning Assignment 2 | Interactive Model Benchmark</div>", unsafe_allow_html=True)

# Helper function to load artifacts
@st.cache_resource
def load_model_artifacts():
    models = {}
    model_files = {
        "Logistic Regression": "model/logistic_regression.pkl",
        "Decision Tree": "model/decision_tree.pkl",
        "kNN": "model/knn.pkl",
        "Naive Bayes": "model/naive_bayes.pkl",
        "Random Forest (Ensemble)": "model/random_forest.pkl"
    }
    
    for name, path in model_files.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
            
    scaler = joblib.load("model/scaler.pkl") if os.path.exists("model/scaler.pkl") else None
    encoders = joblib.load("model/encoders.pkl") if os.path.exists("model/encoders.pkl") else {}
    feature_names = joblib.load("model/feature_names.pkl") if os.path.exists("model/feature_names.pkl") else []
    
    return models, scaler, encoders, feature_names

models, scaler, encoders, feature_names = load_model_artifacts()

# Sidebar Setup
st.sidebar.header("⚙️ App Controls & Data Upload")
st.sidebar.markdown("---")

# Feature a: Dataset Upload Option (CSV)
uploaded_file = st.sidebar.file_uploader("Upload Test Dataset (CSV)", type=["csv"])

@st.cache_data
def load_test_data(file):
    if file is not None:
        df = pd.read_csv(file)
    elif os.path.exists("test_data.csv"):
        df = pd.read_csv("test_data.csv")
    else:
        st.error("No dataset available. Please run model training or upload a test CSV.")
        return None
    return df

df_test = load_test_data(uploaded_file)

if df_test is not None:
    st.sidebar.success(f"Loaded Test Dataset ({df_test.shape[0]} samples, {df_test.shape[1]} columns)")

    # Identify target column
    target_candidates = ['Target', 'y', 'target', 'deposit']
    target_col = None
    for cand in target_candidates:
        if cand in df_test.columns:
            target_col = cand
            break
    if target_col is None:
        target_col = df_test.columns[-1]

    X_test_df = df_test.drop(columns=[target_col])
    y_test_raw = df_test[target_col]

    # Convert y_test to numeric if encoded string
    if y_test_raw.dtype == 'object':
        y_test = (y_test_raw == 'yes').astype(int)
    else:
        y_test = y_test_raw.values

    # Feature b: Model Selection Dropdown
    st.sidebar.markdown("---")
    model_option = st.sidebar.selectbox(
        "Select Classification Model",
        ["All Models Comparison"] + list(models.keys())
    )

    # Tabs layout
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Model Metrics & Evaluation", 
        "📈 Visual Diagnostics (Confusion Matrix & ROC)", 
        "🔮 Real-time Prediction Engine",
        "📄 Dataset & Requirements Overview"
    ])

    # Pre-scale X_test for models requiring scaled input
    if scaler is not None:
        X_test_scaled = scaler.transform(X_test_df)
    else:
        X_test_scaled = X_test_df.values

    # Evaluate metrics helper
    def get_metrics(model_name, model_obj):
        if model_name in ["Logistic Regression", "kNN", "Naive Bayes"]:
            y_pred = model_obj.predict(X_test_scaled)
            y_prob = model_obj.predict_proba(X_test_scaled)[:, 1] if hasattr(model_obj, "predict_proba") else y_pred
        else:
            y_pred = model_obj.predict(X_test_df)
            y_prob = model_obj.predict_proba(X_test_df)[:, 1] if hasattr(model_obj, "predict_proba") else y_pred

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)

        return {
            "Accuracy": acc, "AUC": auc, "Precision": prec,
            "Recall": rec, "F1": f1, "MCC": mcc,
            "y_pred": y_pred, "y_prob": y_prob
        }

    # TAB 1: Evaluation Metrics & Comparison Table
    with tab1:
        st.subheader("Model Performance Metrics")
        
        if model_option == "All Models Comparison":
            summary_data = []
            for name, model_obj in models.items():
                res = get_metrics(name, model_obj)
                summary_data.append({
                    "ML Model Name": name,
                    "Accuracy": f"{res['Accuracy']:.4f}",
                    "AUC": f"{res['AUC']:.4f}",
                    "Precision": f"{res['Precision']:.4f}",
                    "Recall": f"{res['Recall']:.4f}",
                    "F1": f"{res['F1']:.4f}",
                    "MCC": f"{res['MCC']:.4f}"
                })
            
            res_df = pd.DataFrame(summary_data)
            st.dataframe(res_df, use_container_width=True)

            st.markdown("### 🏆 Comprehensive Metrics Comparison Chart")
            df_chart = pd.DataFrame([
                {**{"Model": name}, **{k: float(v) for k, v in row.items() if k != "ML Model Name"}}
                for name, row in zip(models.keys(), summary_data)
            ]).set_index("Model")
            
            st.bar_chart(df_chart)

        else:
            # Single model view
            model_obj = models[model_option]
            res = get_metrics(model_option, model_obj)

            # Display metric cards
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            metrics_list = [
                ("Accuracy", res["Accuracy"], c1),
                ("AUC Score", res["AUC"], c2),
                ("Precision", res["Precision"], c3),
                ("Recall", res["Recall"], c4),
                ("F1 Score", res["F1"], c5),
                ("MCC Score", res["MCC"], c6),
            ]
            for lbl, val, col in metrics_list:
                with col:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-val'>{val:.4f}</div>
                        <div class='metric-lbl'>{lbl}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")
            st.subheader(f"Detailed Classification Report: {model_option}")
            report = classification_report(y_test, res["y_pred"], output_dict=True, target_names=["No Deposit", "Subscribed"])
            st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

    # TAB 2: Visual Diagnostics
    with tab2:
        st.subheader("Confusion Matrix & ROC Curves")
        col_cm, col_roc = st.columns(2)

        if model_option == "All Models Comparison":
            selected_eval_model = st.selectbox("Select Model for Visual Inspection", list(models.keys()))
        else:
            selected_eval_model = model_option

        res_eval = get_metrics(selected_eval_model, models[selected_eval_model])

        with col_cm:
            st.markdown(f"#### Confusion Matrix: `{selected_eval_model}`")
            cm = confusion_matrix(y_test, res_eval["y_pred"])
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                        xticklabels=["No Deposit (0)", "Subscribed (1)"],
                        yticklabels=["No Deposit (0)", "Subscribed (1)"])
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            st.pyplot(fig)

        with col_roc:
            st.markdown("#### ROC Curve Comparison")
            fig_roc, ax_roc = plt.subplots(figsize=(6, 4.5))
            for name, model_obj in models.items():
                m_res = get_metrics(name, model_obj)
                fpr, tpr, _ = roc_curve(y_test, m_res["y_prob"])
                ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {m_res['AUC']:.3f})")
            ax_roc.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.legend(loc="lower right", fontsize=8)
            ax_roc.grid(True, alpha=0.3)
            st.pyplot(fig_roc)

    # TAB 3: Real-time Interactive Predictor
    with tab3:
        st.subheader("🔮 Predict Deposit Subscription for New Client")
        st.write("Adjust client details below to compute real-time subscription probability across models.")

        p_col1, p_col2, p_col3 = st.columns(3)
        
        with p_col1:
            age = st.number_input("Age", 18, 100, 35)
            job = st.selectbox("Job Type", ['admin.', 'blue-collar', 'technician', 'services', 'management', 'retired', 'self-employed'])
            marital = st.selectbox("Marital Status", ['married', 'single', 'divorced'])
            education = st.selectbox("Education Level", ['primary', 'secondary', 'tertiary', 'unknown'])
            default = st.selectbox("Credit in Default?", ['no', 'yes'])
            balance = st.number_input("Average Yearly Balance (€)", -2000, 50000, 1500)

        with p_col2:
            housing = st.selectbox("Housing Loan?", ['no', 'yes'])
            loan = st.selectbox("Personal Loan?", ['no', 'yes'])
            contact = st.selectbox("Contact Communication Type", ['cellular', 'telephone', 'unknown'])
            day = st.slider("Last Contact Day of Month", 1, 31, 15)
            month = st.selectbox("Last Contact Month", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])

        with p_col3:
            duration = st.number_input("Last Contact Duration (seconds)", 0, 4000, 300)
            campaign = st.number_input("Contacts during Campaign", 1, 50, 2)
            pdays = st.number_input("Days since last contact (-1 = never)", -1, 1000, -1)
            previous = st.number_input("Number of contacts before campaign", 0, 50, 0)
            poutcome = st.selectbox("Outcome of Previous Campaign", ['unknown', 'failure', 'other', 'success'])

        # Input mapping
        input_data = pd.DataFrame([{
            'age': age, 'job': job, 'marital': marital, 'education': education,
            'default': default, 'balance': balance, 'housing': housing, 'loan': loan,
            'contact': contact, 'day': day, 'month': month, 'duration': duration,
            'campaign': campaign, 'pdays': pdays, 'previous': previous, 'poutcome': poutcome
        }])

        # Encode input_data matching training encoders
        input_encoded = input_data.copy()
        for col in feature_names:
            if col in encoders:
                le = encoders[col]
                val = str(input_encoded[col].iloc[0])
                if val in le.classes_:
                    input_encoded[col] = le.transform([val])[0]
                else:
                    input_encoded[col] = 0

        st.markdown("---")
        if st.button("🚀 Generate Model Predictions"):
            pred_model = models["Random Forest (Ensemble)" if model_option == "All Models Comparison" else model_option]
            
            if model_option in ["Logistic Regression", "kNN", "Naive Bayes"] or (model_option == "All Models Comparison"):
                input_scaled = scaler.transform(input_encoded)
                prob = pred_model.predict_proba(input_scaled)[0, 1]
            else:
                prob = pred_model.predict_proba(input_encoded)[0, 1]

            sub_pred = "YES (Subscribed)" if prob > 0.5 else "NO (Will Not Subscribe)"
            
            st.markdown(f"### Prediction Result: **{sub_pred}**")
            st.progress(float(prob))
            st.write(f"Estimated Subscription Probability: **{prob*100:.2f}%**")

    # TAB 4: Dataset & Requirements Checklist
    with tab4:
        st.subheader("📋 Assignment Checklist & Dataset Overview")
        st.markdown("""
        - **Dataset Source**: UCI Bank Marketing Repository
        - **Feature Count**: 16 features (Requirement: >= 12)
        - **Instance Count**: 2,000 / 45,211 rows (Requirement: >= 500)
        - **Evaluation Metrics Implemented**: Accuracy, AUC Score, Precision, Recall, F1 Score, MCC Score
        - **Models Implemented**:
          1. Logistic Regression
          2. Decision Tree Classifier
          3. K-Nearest Neighbor Classifier
          4. Naive Bayes Classifier
          5. Random Forest Classifier (Ensemble)
        """)
        st.write("Preview of loaded dataset:")
        st.dataframe(df_test.head(10), use_container_width=True)
