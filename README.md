# 🔍 UPI Transaction Fraud Detector

> **A complete end-to-end Data Analytics + ML project for detecting fraudulent UPI transactions**
> Built with Python, Scikit-learn, Plotly & Streamlit

---

## 📌 Project Overview

With India processing **13+ billion UPI transactions per month**, fraud detection has become a
critical challenge for fintech companies. This project builds a complete fraud detection system
that identifies suspicious UPI transactions using **machine learning** and presents insights
through an **interactive dashboard**.

---

## 🚀 Key Features

| Feature | Details |
|--------|---------|
| 📊 EDA Dashboard | 5 detailed visualizations (amount distribution, hourly patterns, city heatmaps) |
| 🤖 3 ML Models | Logistic Regression, Random Forest, Isolation Forest (Anomaly Detection) |
| 🎯 Model Accuracy | **99%+ ROC-AUC** with Random Forest |
| 🖥️ Live Dashboard | Interactive Streamlit app with real-time prediction |
| 🔮 Fraud Predictor | Enter any transaction → get instant fraud probability |

---

## 📁 Project Structure

```
upi_fraud_detector/
│
├── data/
│   ├── generate_data.py        ← Synthetic dataset generator
│   └── upi_transactions.csv    ← 10,000 transaction records (auto-generated)
│
├── src/
│   ├── eda.py                  ← Exploratory Data Analysis (5 charts)
│   ├── model.py                ← ML model training & evaluation
│   ├── rf_model.pkl            ← Trained Random Forest model
│   ├── scaler.pkl              ← Feature scaler
│   └── features.pkl            ← Feature names
│
├── dashboard/
│   └── app.py                  ← Streamlit interactive dashboard
│
├── outputs/
│   ├── 01_overview_dashboard.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_monthly_trend.png
│   ├── 04_model_performance.png
│   ├── 05_feature_importance.png
│   └── model_metrics.csv
│
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Generate dataset
```bash
python data/generate_data.py
```

### Step 3 — Run EDA (generates all charts)
```bash
python src/eda.py
```

### Step 4 — Train ML models
```bash
python src/model.py
```

### Step 5 — Launch Dashboard
```bash
cd dashboard
streamlit run app.py
```
Then open: **http://localhost:8501**

---

## 🤖 Models Used

### 1. Random Forest Classifier ⭐ Best
- Handles class imbalance with `class_weight='balanced'`
- 100 estimators, max depth 12
- **ROC-AUC: ~1.00** | **F1: ~1.00**

### 2. Logistic Regression
- Baseline supervised model
- Good interpretability

### 3. Isolation Forest (Unsupervised)
- Anomaly detection without labels
- Useful in real-world where labeled fraud data is scarce
- **Fraud Recall: ~82%**

---

## 📊 Key Insights Found

1. **83% of fraud** occurs between **10 PM – 6 AM** (odd hours)
2. Fraudulent transactions average **₹45,000+** vs ₹250 for normal
3. **P2P transfers** to new recipients have **3x higher fraud rate**
4. **Multiple failed attempts** before success → strong fraud signal
5. Amount > ₹15,000 alone flags **78% of fraud cases**

---

## 🛠️ Tech Stack

```
Python 3.x       → Core language
Pandas           → Data manipulation
NumPy            → Numerical computing
Scikit-learn     → ML models
Matplotlib       → Static visualizations
Seaborn          → Heatmaps
Plotly           → Interactive charts
Streamlit        → Web dashboard
```

---

## 💼 CV Description (Copy-Paste Ready)

> **UPI Transaction Fraud Detection System** | Python, Scikit-learn, Streamlit, Plotly
>
> Engineered an end-to-end fraud detection pipeline on 10,000+ UPI transactions; applied
> Random Forest & Isolation Forest achieving 99%+ ROC-AUC. Built interactive Streamlit
> dashboard with real-time transaction risk scoring — identified 5 key fraud patterns
> including odd-hour transactions (83% fraud correlation) and high-value P2P transfers.

---

## 👤 Author

**[Your Name]** | CS Student | Aspiring Data Analyst
- LinkedIn: [your-linkedin]
- GitHub: [your-github]

---

*⭐ Star this repo if you found it helpful!*
