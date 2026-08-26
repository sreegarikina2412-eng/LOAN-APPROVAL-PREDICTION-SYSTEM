"""
cli_app.py
Interactive Console-Based Interface for Loan Approval Prediction.
Includes robust input validation, range checking, and risk factor diagnostics.
"""

import sys
import os

# Add src to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from predict import LoanPredictor


def print_banner():
    banner = """
================================================================================
           LOAN APPROVAL PREDICTION SYSTEM (IBI ML TASK)
================================================================================
  Powered by Machine Learning Classification & Financial Risk Underwriting
================================================================================
"""
    print(banner)


def get_valid_float(prompt: str, min_val: float, max_val: float, default: float = None) -> float:
    while True:
        default_str = f" [{default}]" if default is not None else ""
        user_val = input(f"  > {prompt}{default_str}: ").strip()
        if not user_val and default is not None:
            return default
        try:
            val = float(user_val)
            if min_val <= val <= max_val:
                return val
            print(f"    [!] Value must be between {min_val:,.0f} and {max_val:,.0f}. Please try again.")
        except ValueError:
            print("    [!] Invalid numeric input. Please enter a valid number.")


def get_valid_int(prompt: str, min_val: int, max_val: int, default: int = None) -> int:
    while True:
        default_str = f" [{default}]" if default is not None else ""
        user_val = input(f"  > {prompt}{default_str}: ").strip()
        if not user_val and default is not None:
            return default
        try:
            val = int(user_val)
            if min_val <= val <= max_val:
                return val
            print(f"    [!] Value must be an integer between {min_val} and {max_val}. Please try again.")
        except ValueError:
            print("    [!] Invalid integer input. Please enter a whole number.")


def get_valid_choice(prompt: str, choices: list, default: str = None) -> str:
    choices_lower = [c.lower() for c in choices]
    options_str = "/".join(choices)
    while True:
        default_str = f" [{default}]" if default is not None else ""
        user_val = input(f"  > {prompt} ({options_str}){default_str}: ").strip().lower()
        if not user_val and default is not None:
            return default
        
        # Check matching
        for i, choice in enumerate(choices_lower):
            if user_val == choice or (len(user_val) == 1 and user_val == choice[0]):
                return choices[i]
        
        print(f"    [!] Invalid option. Please choose from: {', '.join(choices)}")


def run_interactive():
    print_banner()
    
    try:
        predictor = LoanPredictor()
    except Exception as e:
        print(f"\n[ERROR] Failed to load model artifacts: {e}")
        print("Please train the model first by running: python src/train.py\n")
        return

    print("Please enter applicant details to evaluate loan eligibility:\n")

    # Collect and validate all inputs
    gender = get_valid_choice("Gender", ["Male", "Female"], default="Male")
    age = get_valid_int("Age (years)", 18, 100, default=35)
    monthly_income = get_valid_float("Monthly Income ($)", 500, 100000, default=5500)
    loan_amount = get_valid_float("Requested Loan Amount ($)", 1000, 1000000, default=120000)
    credit_score = get_valid_int("Credit Score (FICO 300-850)", 300, 850, default=710)
    employment_status = get_valid_choice("Employment Status", ["Employed", "Self-Employed", "Unemployed"], default="Employed")
    existing_loans = get_valid_int("Number of Active Existing Loans", 0, 10, default=0)
    loan_term = get_valid_int("Loan Duration / Term (months, e.g. 12, 36, 60, 120, 180, 240, 360)", 6, 480, default=180)
    property_area = get_valid_choice("Property Area", ["Urban", "Semi-Urban", "Rural"], default="Urban")

    applicant_data = {
        "Gender": gender,
        "Age": age,
        "MonthlyIncome": monthly_income,
        "LoanAmount": loan_amount,
        "CreditScore": credit_score,
        "EmploymentStatus": employment_status,
        "ExistingLoans": existing_loans,
        "LoanTerm": loan_term,
        "PropertyArea": property_area
    }

    # Run Prediction
    print("\n" + "-"*50)
    print("  Processing application with ML Underwriting Engine...")
    print("-"*50)

    result = predictor.predict_single(applicant_data)

    # Format Results Output
    decision_text = ">> LOAN APPROVED <<" if result["is_approved"] else ">> LOAN REJECTED <<"
    border = "=" * 60

    print(f"\n{border}")
    print(f"             PREDICTION ASSESSMENT RESULT")
    print(f"{border}")
    print(f"  Decision:               {decision_text}")
    print(f"  Approval Probability:   {result['approval_probability'] * 100:.2f}%")
    print(f"  Risk Profile:           {result['risk_level']}")
    print(f"  Credit Rating Tier:     {result['credit_tier']}")
    print(f"  Estimated Monthly EMI:  ${result['estimated_emi']:,.2f}")
    print(f"  Debt-to-Income (EMI%):  {result['emi_ratio_percent']:.1f}%")
    print(f"  Loan-to-Income Ratio:   {result['loan_to_income_ratio']:.1f}x monthly income")
    print(f"{border}")

    if result["positives"]:
        print("\n  [+] Key Strengths / Positive Drivers:")
        for pos in result["positives"]:
            print(f"      * {pos}")

    if result["risks"]:
        print("\n  [-] Risk Indicators & Warning Factors:")
        for r in result["risks"]:
            print(f"      * {r}")

    print(f"\n{border}\n")


def run_automated_test():
    """Run non-interactive automated test cases."""
    print("Running automated test cases...")
    predictor = LoanPredictor()

    test_case_approved = {
        "Gender": "Male",
        "Age": 38,
        "MonthlyIncome": 8500,
        "LoanAmount": 120000,
        "CreditScore": 760,
        "EmploymentStatus": "Employed",
        "ExistingLoans": 0,
        "LoanTerm": 180,
        "PropertyArea": "Semi-Urban"
    }

    test_case_rejected = {
        "Gender": "Female",
        "Age": 24,
        "MonthlyIncome": 1200,
        "LoanAmount": 250000,
        "CreditScore": 480,
        "EmploymentStatus": "Unemployed",
        "ExistingLoans": 4,
        "LoanTerm": 36,
        "PropertyArea": "Rural"
    }

    res_app = predictor.predict_single(test_case_approved)
    res_rej = predictor.predict_single(test_case_rejected)

    print(f"Approved Test Case Decision: {res_app['decision']} (Prob: {res_app['approval_probability']:.2%})")
    print(f"Rejected Test Case Decision: {res_rej['decision']} (Prob: {res_rej['approval_probability']:.2%})")
    assert res_app["is_approved"] is True, "Test failed: High credit applicant should be approved"
    assert res_rej["is_approved"] is False, "Test failed: High risk applicant should be rejected"
    print("[SUCCESS] All automated tests passed!")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_automated_test()
    else:
        run_interactive()
