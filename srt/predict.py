"""
predict.py
Inference engine and credit risk explanation logic for loan approval prediction.
"""

import os
import joblib
import numpy as np
import pandas as pd


class LoanPredictor:
    """
    Loan application inference engine.
    Wraps the trained preprocessor and ML classification model.
    """

    def __init__(self, models_dir: str = None):
        if models_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_dir = os.path.dirname(current_dir)
            models_dir = os.path.join(project_dir, "models")

        self.models_dir = models_dir
        self.preprocessor_path = os.path.join(models_dir, "preprocessor.pkl")
        self.model_path = os.path.join(models_dir, "best_model.pkl")

        if not os.path.exists(self.preprocessor_path) or not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model artifacts not found in {models_dir}. Please run 'python src/train.py' first."
            )

        self.preprocessor = joblib.load(self.preprocessor_path)
        self.model = joblib.load(self.model_path)

    def predict_single(self, applicant_data: dict) -> dict:
        """
        Predict loan approval for a single applicant record.

        Parameters:
        -----------
        applicant_data : dict
            Dictionary containing applicant features:
            - Age (int)
            - MonthlyIncome (float)
            - LoanAmount (float)
            - CreditScore (int)
            - EmploymentStatus (str: "Employed", "Self-Employed", "Unemployed")
            - ExistingLoans (int)
            - LoanTerm (int)
            - PropertyArea (str: "Urban", "Semi-Urban", "Rural")
            - Gender (str: "Male", "Female")

        Returns:
        --------
        dict
            Prediction results with probability, approval decision, and risk metrics.
        """
        # Ensure default Gender if not specified
        if "Gender" not in applicant_data:
            applicant_data["Gender"] = "Male"

        df_input = pd.DataFrame([applicant_data])
        X_trans = self.preprocessor.transform(df_input)

        prediction = int(self.model.predict(X_trans)[0])
        probabilities = self.model.predict_proba(X_trans)[0]
        approval_prob = float(probabilities[1])

        # Financial Health Metrics Calculation
        monthly_income = float(applicant_data.get("MonthlyIncome", 0))
        loan_amount = float(applicant_data.get("LoanAmount", 0))
        loan_term = max(int(applicant_data.get("LoanTerm", 1)), 1)
        credit_score = int(applicant_data.get("CreditScore", 300))
        emp_status = applicant_data.get("EmploymentStatus", "")
        existing_loans = int(applicant_data.get("ExistingLoans", 0))

        estimated_emi = loan_amount / loan_term
        emi_ratio = (estimated_emi / monthly_income) * 100 if monthly_income > 0 else 999.0
        loan_to_income = (loan_amount / monthly_income) if monthly_income > 0 else 999.0

        # Credit Score Tier
        if credit_score >= 750:
            credit_tier = "Excellent (750-850)"
        elif credit_score >= 700:
            credit_tier = "Good (700-749)"
        elif credit_score >= 640:
            credit_tier = "Fair (640-699)"
        elif credit_score >= 580:
            credit_tier = "Poor (580-639)"
        else:
            credit_tier = "Very Poor (<580)"

        # Risk Analysis Explanation
        positives = []
        risks = []

        if credit_score >= 700:
            positives.append(f"Strong Credit Score ({credit_score}) indicates high creditworthiness.")
        elif credit_score < 600:
            risks.append(f"Low Credit Score ({credit_score}) significantly increases default risk.")

        if emi_ratio <= 35:
            positives.append(f"Healthy Debt-to-Income / EMI ratio ({emi_ratio:.1f}% of income).")
        elif emi_ratio > 50:
            risks.append(f"High Debt-to-Income / EMI burden ({emi_ratio:.1f}% of income exceeds safe 40% cap).")

        if emp_status == "Employed":
            positives.append("Stable formal employment status.")
        elif emp_status == "Unemployed":
            risks.append("Lack of steady employment or income verification.")

        if existing_loans == 0:
            positives.append("No active prior loans.")
        elif existing_loans >= 3:
            risks.append(f"High number of active existing loans ({existing_loans}).")

        # Overall Risk Rating
        if approval_prob >= 0.75:
            risk_level = "Low Risk"
        elif approval_prob >= 0.50:
            risk_level = "Moderate Risk"
        else:
            risk_level = "High Risk"

        return {
            "is_approved": bool(prediction == 1),
            "decision": "APPROVED" if prediction == 1 else "REJECTED",
            "approval_probability": approval_prob,
            "rejection_probability": float(probabilities[0]),
            "risk_level": risk_level,
            "credit_tier": credit_tier,
            "estimated_emi": estimated_emi,
            "emi_ratio_percent": emi_ratio,
            "loan_to_income_ratio": loan_to_income,
            "positives": positives,
            "risks": risks
        }
