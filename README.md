# 💳 Loan Approval Prediction System
### Machine Learning Development Internship Task — Infobharat Interns (IBI)

![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)

---

## 📌 Project Overview
The **Loan Approval Prediction System** is an end-to-end Machine Learning pipeline and credit risk assessment tool developed to automate and optimize the retail loan underwriting process. By analyzing applicant demographics, financial health, employment profile, and requested loan terms, the system predicts whether a loan application should be **Approved (1)** or **Rejected (0)** with high accuracy and transparent risk diagnostics.

This project covers the complete machine learning lifecycle:
1. **Realistic Synthetic Data Generation** ($\ge 2,000$ applicant records with banking credit rules).
2. **Robust Data Preprocessing & Feature Engineering** (Debt-to-Income / EMI leverage metrics, One-Hot Encoding, StandardScaler).
3. **Exploratory Data Analysis (EDA) & Data Visualizations** (Matplotlib & Seaborn).
4. **Machine Learning Model Benchmarking** (Logistic Regression, Decision Tree, Random Forest).
5. **Interactive Console Prediction CLI** with comprehensive input validation.
6. **Bonus Interactive Streamlit Web Application** with live probability gauges, risk diagnostics, and What-If analysis.

---

## 🗂️ Project Directory Structure

```
loan-approval-prediction-system/
├── data/
│   └── loan_approval_dataset.csv          # Synthetic dataset (2,000 applicant records)
├── models/
│   ├── best_model.pkl                     # Serialized best model artifact (Random Forest)
│   └── preprocessor.pkl                   # Reusable preprocessing pipeline
├── notebooks/
│   └── loan_approval_prediction.ipynb     # Complete Jupyter Notebook walkthrough
├── src/
│   ├── __init__.py
│   ├── data_generator.py                  # Synthetic data creation with underwriting logic
│   ├── preprocessing.py                   # Data cleaning, imputation, encoding, & scaling
│   ├── train.py                           # Model training, CV evaluation & chart export
│   └── predict.py                         # Core inference & risk explanation engine
├── visualizations/
│   ├── employment_approval_bar.png        # Approvals by employment status
│   ├── correlation_heatmap.png            # Financial feature correlation matrix
│   ├── income_vs_loanamount_scatter.png   # Monthly income vs loan amount scatter plot
│   ├── approval_distribution_pie.png      # Class distribution pie chart
│   ├── model_comparison_bar.png           # Multi-model evaluation benchmark
│   ├── confusion_matrix_best.png          # Confusion matrix for Random Forest
│   └── feature_importance.png             # Top 10 predictive features
├── app.py                                 # Interactive Streamlit Web Application
├── cli_app.py                             # Interactive Console Prediction Interface
├── requirements.txt                       # Python dependencies
└── README.md                              # Project documentation & submission report
```

---

## 📊 Dataset Schema & Generation

The dataset contains **2,000 records** generated using NumPy and Pandas with realistic banking underwriting constraints:

| Feature Name | Type | Description | Values / Range |
| :--- | :--- | :--- | :--- |
| **`ApplicantID`** | String | Unique applicant code | `LP001001` - `LP003000` |
| **`Gender`** | Categorical | Gender of applicant | `Male`, `Female` |
| **`Age`** | Integer | Age in years | `21` - `68` |
| **`MonthlyIncome`** | Float | Monthly income ($) | `$800` - `$35,000` |
| **`LoanAmount`** | Float | Requested principal amount ($) | `$5,000` - `$500,000` |
| **`CreditScore`** | Integer | FICO credit score | `300` - `850` |
| **`EmploymentStatus`**| Categorical | Employment category | `Employed`, `Self-Employed`, `Unemployed` |
| **`ExistingLoans`** | Integer | Active loan count | `0` - `5` |
| **`LoanTerm`** | Integer | Loan duration in months | `12, 36, 60, 120, 180, 240, 360` |
| **`PropertyArea`** | Categorical | Geographic classification | `Urban`, `Semi-Urban`, `Rural` |
| **`LoanApproved`** | Binary Target | Ground truth approval decision | `1` (Approved), `0` (Rejected) |

---

## 📈 Exploratory Data Analysis & Visualizations

| Visual Chart | Preview / Path | Key Analytical Insight |
| :--- | :--- | :--- |
| **Approval by Employment** | `visualizations/employment_approval_bar.png` | Formally employed applicants have the highest approval rate (~88%), while unemployed applicants face high rejection rates due to lack of verified income. |
| **Feature Correlation** | `visualizations/correlation_heatmap.png` | `CreditScore` shows the strongest positive correlation with approval (+0.58), while high `ExistingLoans` and high loan amounts relative to income show negative correlation. |
| **Income vs. Loan Amount** | `visualizations/income_vs_loanamount_scatter.png` | Approval is heavily gated by the Debt-to-Income / EMI ratio. Higher loan amounts require proportionally higher incomes. |
| **Class Distribution** | `visualizations/approval_distribution_pie.png` | Healthy distribution (~77% approved, ~23% rejected) representative of typical retail loan portfolio approvals. |

---

## 🤖 Machine Learning Model Benchmarking

We benchmarked three classification algorithms using **Stratified 5-Fold Cross-Validation** and an 80/20 train-test split:

| Model | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC | 5-Fold CV Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest Classifier** 🏆 | **91.75%** | **92.59%** | **97.09%** | **0.9479** | **0.9215** | **86.88% (±1.49%)** |
| **Logistic Regression** | 89.50% | 91.85% | 94.82% | 0.9331 | 0.9347 | 86.75% (±1.08%) |
| **Decision Tree Classifier** | 87.25% | 89.09% | 95.15% | 0.9202 | 0.8985 | 86.19% (±2.00%) |

### Top Predictive Feature Importances:
1. **`CreditScore`** (~42%): The primary metric determining repayment probability.
2. **`EMIToIncomeRatio`** (~21%): Monthly payment burden relative to take-home earnings.
3. **`LoanToIncomeRatio`** (~14%): Overall leverage ratio.
4. **`MonthlyIncome`** (~9%): Gross monthly cash flow.
5. **`ExistingLoans`** (~6%): Prior debt obligations.

---

## 🚀 How to Run the Project

### 1. Installation & Environment Setup
Clone the repository and install required packages:
```bash
git clone https://github.com/<your-username>/loan-approval-prediction-system.git
cd loan-approval-prediction-system
pip install -r requirements.txt
```

### 2. Generate Dataset & Train Models
To regenerate the synthetic dataset, create EDA charts, and train all ML models:
```bash
python src/train.py
```

### 3. Run the Interactive Console Application (CLI)
To test predictions in the interactive terminal:
```bash
python cli_app.py
```
*(Supports automated non-interactive testing via `python cli_app.py --test`)*

### 4. Launch the Streamlit Web Application (Bonus)
To launch the modern interactive browser interface:
```bash
streamlit run app.py
```

### 5. Run the Jupyter Notebook
```bash
jupyter notebook notebooks/loan_approval_prediction.ipynb
```

---

## 💡 Key Business & Risk Insights
- **Credit Score is Paramount**: Applicants with credit scores $\ge 700$ experience $>95\%$ approval likelihood, whereas scores $<580$ are rejected in $>85\%$ of cases.
- **The 40% EMI Rule**: If an applicant's estimated monthly EMI exceeds 40% of their monthly income, the default risk increases sharply.
- **Leverage Control**: Having 3 or more existing active loans significantly suppresses approval odds unless offset by substantial income.

---

## 🌐 LinkedIn / Demo Submission Post Template

```markdown
🚀 Excited to share my latest Machine Learning project: **Loan Approval Prediction System** developed as part of the **Infobharat Interns (IBI)** Machine Learning Internship! 💳📊

In this project, I built an end-to-end classification pipeline to automate retail loan underwriting decisions and risk diagnostics:

🔹 **Dataset Architecture:** Created a synthetic dataset of 2,000+ records enforcing realistic banking underwriting constraints (Credit Score tiers, Debt-to-Income ratios, Employment stability).
🔹 **Data Preprocessing & Feature Engineering:** Engineered custom financial risk metrics (`EMIToIncomeRatio`, `LoanToIncomeRatio`), handled missing values, and built reusable `ColumnTransformer` pipelines.
🔹 **Exploratory Data Analysis:** Extracted multi-dimensional insights across credit tiers, income distribution, and property classifications using Matplotlib & Seaborn.
🔹 **Model Benchmarking:** Evaluated Logistic Regression, Decision Tree, and Random Forest models across 5-Fold Stratified Cross-Validation. Random Forest achieved **91.75% Accuracy** and **0.948 F1-Score**.
🔹 **Interactive Interfaces:** Developed both a robust Console CLI and an interactive **Streamlit Web Application** featuring real-time risk gauges, sensitivity analysis, and automated financial diagnostics.

🔗 **GitHub Repository:** [Insert your repo link here]

A huge thank you to **Infobharat Interns** for this practical learning assignment!

#MachineLearning #DataScience #Python #ScikitLearn #Streamlit #Fintech #CreditRisk #AI #InfobharatInterns #InternshipProject
```

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
