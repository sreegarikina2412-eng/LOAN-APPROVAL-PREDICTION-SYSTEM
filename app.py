"""
app.py
Interactive Streamlit Web Application for Loan Approval Prediction System.
Includes:
- Single Applicant Real-Time Evaluator with Risk Diagnostics
- Batch CSV File Upload & Processing with Automated Underwriting
- Downloadable Predictions & Sample CSV Template
- Exploratory Data Analysis & Visualizations Gallery
- Model Performance & Feature Importance Insights
"""

import os
import sys
import io
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from predict import LoanPredictor

st.set_page_config(
    page_title="Loan Approval Prediction System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 20px;
    }
    .approved-badge {
        background-color: #DEF7EC;
        color: #03543F;
        font-size: 1.8rem;
        font-weight: 700;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        border: 2px solid #31C48D;
    }
    .rejected-badge {
        background-color: #FDE8E8;
        color: #9B1C1C;
        font-size: 1.8rem;
        font-weight: 700;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        border: 2px solid #F98080;
    }
    .metric-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_predictor():
    return LoanPredictor()


@st.cache_data
def load_dataset():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "loan_approval_dataset.csv")
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None


def generate_sample_csv_template():
    sample_df = pd.DataFrame([
        {
            "ApplicantID": "LP009001", "Gender": "Male", "Age": 35,
            "MonthlyIncome": 8500, "LoanAmount": 120000, "CreditScore": 760,
            "EmploymentStatus": "Employed", "ExistingLoans": 0,
            "LoanTerm": 180, "PropertyArea": "Urban"
        },
        {
            "ApplicantID": "LP009002", "Gender": "Female", "Age": 42,
            "MonthlyIncome": 6200, "LoanAmount": 150000, "CreditScore": 680,
            "EmploymentStatus": "Self-Employed", "ExistingLoans": 1,
            "LoanTerm": 240, "PropertyArea": "Semi-Urban"
        },
        {
            "ApplicantID": "LP009003", "Gender": "Male", "Age": 24,
            "MonthlyIncome": 1200, "LoanAmount": 200000, "CreditScore": 490,
            "EmploymentStatus": "Unemployed", "ExistingLoans": 4,
            "LoanTerm": 36, "PropertyArea": "Rural"
        },
        {
            "ApplicantID": "LP009004", "Gender": "Female", "Age": 29,
            "MonthlyIncome": 4800, "LoanAmount": 80000, "CreditScore": 710,
            "EmploymentStatus": "Employed", "ExistingLoans": 1,
            "LoanTerm": 120, "PropertyArea": "Urban"
        },
        {
            "ApplicantID": "LP009005", "Gender": "Male", "Age": 55,
            "MonthlyIncome": 12000, "LoanAmount": 300000, "CreditScore": 820,
            "EmploymentStatus": "Employed", "ExistingLoans": 2,
            "LoanTerm": 360, "PropertyArea": "Semi-Urban"
        }
    ])
    return sample_df


def process_batch_dataframe(df: pd.DataFrame, predictor: LoanPredictor) -> pd.DataFrame:
    df_out = df.copy()
    
    # Standardize column names (strip whitespace)
    df_out.columns = df_out.columns.str.strip()

    predictions = []
    probabilities = []
    risk_levels = []
    credit_tiers = []
    emis = []
    emi_ratios = []

    for idx, row in df_out.iterrows():
        applicant_dict = row.to_dict()
        try:
            res = predictor.predict_single(applicant_dict)
            predictions.append(res["decision"])
            probabilities.append(round(res["approval_probability"] * 100, 2))
            risk_levels.append(res["risk_level"])
            credit_tiers.append(res["credit_tier"])
            emis.append(round(res["estimated_emi"], 2))
            emi_ratios.append(round(res["emi_ratio_percent"], 2))
        except Exception as e:
            predictions.append("ERROR")
            probabilities.append(0.0)
            risk_levels.append("Unknown")
            credit_tiers.append("Unknown")
            emis.append(0.0)
            emi_ratios.append(0.0)

    df_out["Predicted_Decision"] = predictions
    df_out["Approval_Probability_%"] = probabilities
    df_out["Risk_Level"] = risk_levels
    df_out["Credit_Tier"] = credit_tiers
    df_out["Estimated_Monthly_EMI_$"] = emis
    df_out["Debt_To_Income_Ratio_%"] = emi_ratios

    return df_out


def main():
    st.markdown('<div class="main-header">💳 Loan Approval Prediction System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Automated Financial Underwriting & Batch Risk Assessment Engine (IBI Internship Task)</div>', unsafe_allow_html=True)

    tab_eval, tab_upload, tab_eda, tab_model = st.tabs([
        "🎯 Single Applicant Evaluator",
        "📁 Batch CSV Upload & Underwriting",
        "📊 Exploratory Data Analysis",
        "🤖 Model Performance & Insights"
    ])

    # =============================================================
    # TAB 1: Single Applicant Real-Time Evaluator
    # =============================================================
    with tab_eval:
        st.sidebar.header("📋 Single Applicant Profile")

        gender = st.sidebar.selectbox("Gender", ["Male", "Female"], index=0)
        age = st.sidebar.slider("Age (Years)", min_value=18, max_value=75, value=35)
        monthly_income = st.sidebar.number_input("Monthly Income ($)", min_value=500.0, max_value=100000.0, value=6500.0, step=250.0)
        loan_amount = st.sidebar.number_input("Requested Loan Amount ($)", min_value=1000.0, max_value=1000000.0, value=120000.0, step=5000.0)
        credit_score = st.sidebar.slider("Credit Score (FICO 300 - 850)", min_value=300, max_value=850, value=720, step=5)
        employment_status = st.sidebar.selectbox("Employment Status", ["Employed", "Self-Employed", "Unemployed"], index=0)
        existing_loans = st.sidebar.selectbox("Active Existing Loans", [0, 1, 2, 3, 4, 5], index=0)
        loan_term = st.sidebar.selectbox("Loan Duration (Months)", [12, 36, 60, 120, 180, 240, 360], index=4)
        property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semi-Urban", "Rural"], index=1)

        applicant_data = {
            "Gender": gender, "Age": age, "MonthlyIncome": monthly_income,
            "LoanAmount": loan_amount, "CreditScore": credit_score,
            "EmploymentStatus": employment_status, "ExistingLoans": existing_loans,
            "LoanTerm": loan_term, "PropertyArea": property_area
        }

        try:
            predictor = get_predictor()
            result = predictor.predict_single(applicant_data)
        except Exception as e:
            st.error(f"Error loading model: {e}. Please run `python src/train.py` first.")
            return

        col_decision, col_prob = st.columns([1.2, 1])

        with col_decision:
            if result["is_approved"]:
                st.markdown(
                    f'<div class="approved-badge">✅ LOAN APPROVED<br><span style="font-size:1.1rem;font-weight:normal;">Approval Confidence: {result["approval_probability"]*100:.1f}%</span></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="rejected-badge">❌ LOAN REJECTED<br><span style="font-size:1.1rem;font-weight:normal;">Rejection Confidence: {result["rejection_probability"]*100:.1f}%</span></div>',
                    unsafe_allow_html=True
                )

        with col_prob:
            st.write(f"**Risk Rating:** `{result['risk_level']}` | **Credit Tier:** `{result['credit_tier']}`")
            st.progress(result["approval_probability"], text=f"Approval Probability: {result['approval_probability']*100:.1f}%")
            if result["approval_probability"] >= 0.7:
                st.success("Applicant demonstrates strong repayment capacity and favorable credit metrics.")
            elif result["approval_probability"] >= 0.5:
                st.warning("Moderate risk tier; manual underwriting review or collateral recommended.")
            else:
                st.error("High default risk identified based on low credit score or high debt burden.")

        st.markdown("---")
        st.subheader("📈 Financial Health & Leverage Metrics")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Est. Monthly EMI", f"${result['estimated_emi']:,.2f}")
        m2.metric("EMI / Income Ratio", f"{result['emi_ratio_percent']:.1f}%", help="Safe threshold is generally below 40%")
        m3.metric("Loan-to-Income", f"{result['loan_to_income_ratio']:.1f}x Income")
        m4.metric("Credit Tier", result["credit_tier"].split(" ")[0])

        st.markdown("---")
        col_pos, col_risk = st.columns(2)

        with col_pos:
            st.markdown("### 🟢 Positive Decision Factors")
            if result["positives"]:
                for pos in result["positives"]:
                    st.write(f"✅ {pos}")
            else:
                st.info("No standout positive drivers recorded.")

        with col_risk:
            st.markdown("### 🔴 Risk Indicators & Warnings")
            if result["risks"]:
                for r in result["risks"]:
                    st.write(f"⚠️ {r}")
            else:
                st.success("No adverse risk indicators identified.")

    # =============================================================
    # TAB 2: Batch CSV Upload & Underwriting
    # =============================================================
    with tab_upload:
        st.subheader("📁 Upload CSV File for Batch Loan Underwriting")
        st.write("Upload a CSV file containing applicant records to evaluate multiple loan applications simultaneously.")

        # Download Template Section
        col_info, col_dl = st.columns([3, 1])
        with col_info:
            st.info("💡 **Expected Columns in CSV:** `Gender`, `Age`, `MonthlyIncome`, `LoanAmount`, `CreditScore`, `EmploymentStatus`, `ExistingLoans`, `LoanTerm`, `PropertyArea` (and optional `ApplicantID`).")
        with col_dl:
            sample_template = generate_sample_csv_template()
            csv_buffer = io.StringIO()
            sample_template.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download Sample Template CSV",
                data=csv_buffer.getvalue(),
                file_name="sample_loan_applicants_template.csv",
                mime="text/csv",
                use_container_width=True
            )

        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], key="batch_csv_uploader")

        if uploaded_file is not None:
            try:
                user_df = pd.read_csv(uploaded_file)
                st.success(f"✅ Successfully loaded **{len(user_df)}** records from `{uploaded_file.name}`.")
                
                with st.expander("Preview Uploaded Data (First 5 Rows)", expanded=False):
                    st.dataframe(user_df.head(), use_container_width=True)

                predictor = get_predictor()
                with st.spinner("Processing applications through ML Underwriting Model..."):
                    results_df = process_batch_dataframe(user_df, predictor)

                # Key Metrics Cards
                total_records = len(results_df)
                approved_count = (results_df["Predicted_Decision"] == "APPROVED").sum()
                rejected_count = (results_df["Predicted_Decision"] == "REJECTED").sum()
                approval_rate = (approved_count / total_records) * 100 if total_records > 0 else 0
                avg_prob = results_df["Approval_Probability_%"].mean()
                total_loan_volume = results_df["LoanAmount"].sum() if "LoanAmount" in results_df.columns else 0

                st.markdown("---")
                st.subheader("📊 Batch Processing Summary Dashboard")

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Total Applicants", f"{total_records:,}")
                c2.metric("Approved", f"{approved_count:,}", f"{approval_rate:.1f}% rate")
                c3.metric("Rejected", f"{rejected_count:,}")
                c4.metric("Avg Approval Prob", f"{avg_prob:.1f}%")
                c5.metric("Total Loan Volume", f"${total_loan_volume:,.0f}")

                # Visual Summary Charts
                st.markdown("---")
                ch_col1, ch_col2 = st.columns(2)

                with ch_col1:
                    fig_pie, ax_pie = plt.subplots(figsize=(6, 4))
                    labels = ["Approved", "Rejected"]
                    sizes = [approved_count, rejected_count]
                    colors = ["#2ECC71", "#E74C3C"]
                    ax_pie.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors, startangle=140, explode=(0.05, 0), shadow=True)
                    ax_pie.set_title("Batch Approval Decision Breakdown", fontweight="bold")
                    st.pyplot(fig_pie)
                    plt.close(fig_pie)

                with ch_col2:
                    if "Risk_Level" in results_df.columns:
                        fig_risk, ax_risk = plt.subplots(figsize=(6, 4))
                        risk_order = ["Low Risk", "Moderate Risk", "High Risk"]
                        sns.countplot(data=results_df, x="Risk_Level", order=risk_order, palette={"Low Risk": "#2ECC71", "Moderate Risk": "#F39C12", "High Risk": "#E74C3C"}, ax=ax_risk)
                        ax_risk.set_title("Applicant Risk Profile Distribution", fontweight="bold")
                        ax_risk.set_xlabel("Risk Tier", fontweight="bold")
                        ax_risk.set_ylabel("Count", fontweight="bold")
                        st.pyplot(fig_risk)
                        plt.close(fig_risk)

                # Filterable Table
                st.markdown("---")
                st.subheader("📋 Detailed Batch Underwriting Results")

                filter_choice = st.radio(
                    "Filter Records:",
                    ["All Applications", "Approved Only", "Rejected Only", "High Risk Only"],
                    horizontal=True
                )

                filtered_df = results_df.copy()
                if filter_choice == "Approved Only":
                    filtered_df = filtered_df[filtered_df["Predicted_Decision"] == "APPROVED"]
                elif filter_choice == "Rejected Only":
                    filtered_df = filtered_df[filtered_df["Predicted_Decision"] == "REJECTED"]
                elif filter_choice == "High Risk Only":
                    filtered_df = filtered_df[filtered_df["Risk_Level"] == "High Risk"]

                st.dataframe(filtered_df, use_container_width=True)

                # Export Download Button
                out_buffer = io.StringIO()
                results_df.to_csv(out_buffer, index=False)

                st.download_button(
                    label="📥 Download Complete Predictions with Diagnostics (CSV)",
                    data=out_buffer.getvalue(),
                    file_name=f"loan_underwriting_results_{uploaded_file.name}",
                    mime="text/csv",
                    use_container_width=True
                )

            except Exception as ex:
                st.error(f"Error processing CSV file: {ex}")

    # =============================================================
    # TAB 3: Exploratory Data Analysis Gallery
    # =============================================================
    with tab_eda:
        st.subheader("Exploratory Data Analysis & Visualizations")
        st.write("Visualizations generated from 2,000 applicant records:")

        v_col1, v_col2 = st.columns(2)
        project_dir = os.path.dirname(os.path.abspath(__file__))
        viz_dir = os.path.join(project_dir, "visualizations")

        with v_col1:
            st.markdown("#### 1. Approvals by Employment Status")
            p1 = os.path.join(viz_dir, "employment_approval_bar.png")
            if os.path.exists(p1):
                st.image(p1, use_container_width=True)

            st.markdown("#### 3. Monthly Income vs. Loan Amount")
            p3 = os.path.join(viz_dir, "income_vs_loanamount_scatter.png")
            if os.path.exists(p3):
                st.image(p3, use_container_width=True)

        with v_col2:
            st.markdown("#### 2. Feature Correlation Heatmap")
            p2 = os.path.join(viz_dir, "correlation_heatmap.png")
            if os.path.exists(p2):
                st.image(p2, use_container_width=True)

            st.markdown("#### 4. Loan Approval Class Distribution")
            p4 = os.path.join(viz_dir, "approval_distribution_pie.png")
            if os.path.exists(p4):
                st.image(p4, use_container_width=True)

        df_preview = load_dataset()
        if df_preview is not None:
            st.markdown("---")
            st.subheader("Raw Dataset Sample (First 10 Rows)")
            st.dataframe(df_preview.head(10), use_container_width=True)

    # =============================================================
    # TAB 4: Model Performance & Insights
    # =============================================================
    with tab_model:
        st.subheader("Machine Learning Model Performance & Diagnostics")

        col_m1, col_m2 = st.columns(2)
        project_dir = os.path.dirname(os.path.abspath(__file__))
        viz_dir = os.path.join(project_dir, "visualizations")

        with col_m1:
            st.markdown("#### Model Comparison Benchmark")
            p_comp = os.path.join(viz_dir, "model_comparison_bar.png")
            if os.path.exists(p_comp):
                st.image(p_comp, use_container_width=True)

        with col_m2:
            st.markdown("#### Random Forest Confusion Matrix")
            p_cm = os.path.join(viz_dir, "confusion_matrix_best.png")
            if os.path.exists(p_cm):
                st.image(p_cm, use_container_width=True)

        st.markdown("#### Top Predictive Feature Importances")
        p_fi = os.path.join(viz_dir, "feature_importance.png")
        if os.path.exists(p_fi):
            st.image(p_fi, use_container_width=True)


if __name__ == "__main__":
    main()
