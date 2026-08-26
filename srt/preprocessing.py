"""
preprocessing.py
Data cleaning, missing value handling, categorical encoding,
feature scaling, and reusable preprocessing pipeline.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


class LoanDataPreprocessor(BaseEstimator, TransformerMixin):
    """
    End-to-end preprocessing pipeline for loan applicant data.
    Performs data validation, missing value imputation, feature engineering,
    one-hot encoding, and feature scaling.
    """

    def __init__(self):
        self.numeric_features = [
            "Age", "MonthlyIncome", "LoanAmount", "CreditScore",
            "ExistingLoans", "LoanTerm", "EMIToIncomeRatio", "LoanToIncomeRatio"
        ]
        self.categorical_features = ["Gender", "EmploymentStatus", "PropertyArea"]
        self.pipeline = None
        self.feature_names_out_ = None
        self.imputer_num_ = None
        self.imputer_cat_ = None

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create domain-specific financial risk ratios."""
        df_copy = df.copy()
        
        # Ensure numeric types
        for col in ["Age", "MonthlyIncome", "LoanAmount", "CreditScore", "ExistingLoans", "LoanTerm"]:
            if col in df_copy.columns:
                df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce")

        # Handle missing income or loan term before ratio calculation
        safe_income = df_copy["MonthlyIncome"].fillna(5000.0).replace(0, 1.0)
        safe_term = df_copy["LoanTerm"].fillna(180.0).replace(0, 12.0)
        safe_loan = df_copy["LoanAmount"].fillna(100000.0)

        # 1. Estimated monthly EMI
        estimated_emi = safe_loan / safe_term
        # 2. EMI to monthly income ratio
        df_copy["EMIToIncomeRatio"] = np.clip(estimated_emi / safe_income, 0.0, 5.0)
        # 3. Loan amount to monthly income ratio
        df_copy["LoanToIncomeRatio"] = np.clip(safe_loan / safe_income, 0.0, 100.0)

        return df_copy

    def fit(self, X: pd.DataFrame, y=None):
        """Fit imputation, encoding, and scaling transformers."""
        X_eng = self._engineer_features(X)

        # Numeric transformer: Median Imputation -> Standard Scaling
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        # Categorical transformer: Most Frequent Imputation -> One-Hot Encoding
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        # Combine into ColumnTransformer
        self.pipeline = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.numeric_features),
                ("cat", categorical_transformer, self.categorical_features)
            ],
            remainder="drop"
        )

        self.pipeline.fit(X_eng, y)

        # Compute output feature names
        cat_encoder = self.pipeline.named_transformers_["cat"].named_steps["encoder"]
        cat_feature_names = list(cat_encoder.get_feature_names_out(self.categorical_features))
        self.feature_names_out_ = self.numeric_features + cat_feature_names

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform applicant dataframe into preprocessed feature matrix."""
        if self.pipeline is None:
            raise RuntimeError("Preprocessor has not been fitted yet. Call fit() first.")
        X_eng = self._engineer_features(X)
        return self.pipeline.transform(X_eng)

    def fit_transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        """Fit and transform applicant data in one step."""
        return self.fit(X, y).transform(X)

    def get_feature_names(self):
        """Return human-readable names of transformed feature columns."""
        return self.feature_names_out_

    def save(self, filepath: str):
        """Serialize preprocessor to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "LoanDataPreprocessor":
        """Load preprocessor from disk."""
        return joblib.load(filepath)


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Initial data cleaning: drop duplicates and handle corrupted rows.
    """
    initial_count = len(df)
    df_cleaned = df.drop_duplicates()
    dropped_dups = initial_count - len(df_cleaned)
    if dropped_dups > 0:
        print(f"[INFO] Removed {dropped_dups} duplicate rows.")

    # Drop ApplicantID if present
    if "ApplicantID" in df_cleaned.columns:
        df_cleaned = df_cleaned.drop(columns=["ApplicantID"])

    return df_cleaned
