"""
train.py
Model training, evaluation, comparison, visualization generation,
and model serialization for Loan Approval Prediction System.
"""

import os
import sys

# Ensure non-GUI backend for Matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# Add src to sys.path if not present
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from preprocessing import LoanDataPreprocessor, clean_raw_data
from data_generator import generate_loan_data

# Set styling for plots
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 300


def create_visualizations(df: pd.DataFrame, output_dir: str):
    """
    Generate and save required and bonus EDA visualizations.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n[INFO] Generating Exploratory Data Visualizations...")

    # 1. Bar chart showing loan approvals by employment status
    plt.figure(figsize=(8, 5))
    emp_df = df.dropna(subset=["EmploymentStatus", "LoanApproved"]).copy()
    emp_df["ApprovalStatus"] = emp_df["LoanApproved"].map({1: "Approved", 0: "Rejected"})
    
    palette = {"Approved": "#2ECC71", "Rejected": "#E74C3C"}
    ax = sns.countplot(
        data=emp_df,
        x="EmploymentStatus",
        hue="ApprovalStatus",
        palette=palette,
        edgecolor="black",
        linewidth=0.8
    )
    plt.title("Loan Approval Distribution by Employment Status", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Employment Status", fontsize=12, fontweight="bold")
    plt.ylabel("Number of Applicants", fontsize=12, fontweight="bold")
    plt.legend(title="Decision", frameon=True)
    
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{int(height)}",
                        (p.get_x() + p.get_width() / 2., height),
                        ha="center", va="bottom", fontsize=10, xytext=(0, 3),
                        textcoords="offset points")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "employment_approval_bar.png"))
    plt.close("all")
    print("  -> Saved employment_approval_bar.png")

    # 2. Correlation Heatmap of Numerical Features
    plt.figure(figsize=(10, 8))
    numeric_cols = ["Age", "MonthlyIncome", "LoanAmount", "CreditScore", "ExistingLoans", "LoanTerm", "LoanApproved"]
    num_df = df[numeric_cols].dropna()
    corr_matrix = num_df.corr()
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    plt.title("Correlation Matrix of Financial & Demographic Features", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
    plt.close("all")
    print("  -> Saved correlation_heatmap.png")

    # 3. Scatter Plot of Income vs Loan Amount (Colored by Approval)
    plt.figure(figsize=(9, 6))
    scatter_df = df.dropna(subset=["MonthlyIncome", "LoanAmount", "LoanApproved"]).copy()
    scatter_df["ApprovalStatus"] = scatter_df["LoanApproved"].map({1: "Approved", 0: "Rejected"})
    
    sns.scatterplot(
        data=scatter_df,
        x="MonthlyIncome",
        y="LoanAmount",
        hue="ApprovalStatus",
        palette=palette,
        alpha=0.7,
        edgecolor="w",
        s=60
    )
    plt.title("Monthly Income vs. Requested Loan Amount by Approval Status", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Monthly Income ($)", fontsize=12, fontweight="bold")
    plt.ylabel("Loan Amount ($)", fontsize=12, fontweight="bold")
    plt.legend(title="Loan Status", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "income_vs_loanamount_scatter.png"))
    plt.close("all")
    print("  -> Saved income_vs_loanamount_scatter.png")

    # 4. Pie Chart of Approved vs Rejected Applications
    plt.figure(figsize=(7, 7))
    approved_counts = df["LoanApproved"].value_counts()
    labels = ["Approved (1)", "Rejected (0)"]
    sizes = [approved_counts.get(1, 0), approved_counts.get(0, 0)]
    colors = ["#2ECC71", "#E74C3C"]
    explode = (0.05, 0)
    
    plt.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        shadow=True,
        startangle=140,
        textprops={"fontsize": 13, "fontweight": "bold"}
    )
    plt.title("Overall Loan Approval Ratio", fontsize=15, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "approval_distribution_pie.png"))
    plt.close("all")
    print("  -> Saved approval_distribution_pie.png")


def train_and_evaluate_models():
    """
    Main pipeline: Data Loading -> Preprocessing -> Model Training -> Evaluation -> Model Export.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    data_dir = os.path.join(project_dir, "data")
    models_dir = os.path.join(project_dir, "models")
    viz_dir = os.path.join(project_dir, "visualizations")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    csv_path = os.path.join(data_dir, "loan_approval_dataset.csv")
    if not os.path.exists(csv_path):
        print(f"[INFO] Dataset not found at {csv_path}. Generating now...")
        df_raw = generate_loan_data(n_samples=2000, random_state=42, inject_missing=True)
        os.makedirs(data_dir, exist_ok=True)
        df_raw.to_csv(csv_path, index=False)
    else:
        print(f"[INFO] Loading dataset from {csv_path}...")
        df_raw = pd.read_csv(csv_path)

    print(f"[INFO] Raw Dataset Shape: {df_raw.shape}")
    
    # Generate EDA Visualizations
    create_visualizations(df_raw, viz_dir)

    # Clean data & separate features/target
    df_cleaned = clean_raw_data(df_raw)
    X = df_cleaned.drop(columns=["LoanApproved"])
    y = df_cleaned["LoanApproved"]

    # Stratified Train-Test Split (80% Train, 20% Test)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"\n[INFO] Training samples: {len(X_train_raw)}, Test samples: {len(X_test_raw)}")

    # Fit Preprocessing Pipeline
    preprocessor = LoanDataPreprocessor()
    X_train = preprocessor.fit_transform(X_train_raw, y_train)
    X_test = preprocessor.transform(X_test_raw)
    feature_names = preprocessor.get_feature_names()

    # Save Preprocessor
    preprocessor_path = os.path.join(models_dir, "preprocessor.pkl")
    preprocessor.save(preprocessor_path)
    print(f"[SUCCESS] Preprocessor saved to {preprocessor_path}")

    # Define Candidate Models
    models = {
        "Logistic Regression": LogisticRegression(
            C=1.0, max_iter=1000, random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, min_samples_split=15, min_samples_leaf=5, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=8, min_samples_split=8, min_samples_leaf=4,
            random_state=42, n_jobs=1
        )
    }

    results = []
    trained_models = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n" + "="*70)
    print("                MODEL BENCHMARKING & EVALUATION")
    print("="*70)

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_prob)

        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": roc,
            "CV Accuracy (Mean)": cv_mean,
            "CV Accuracy (Std)": cv_std
        })

        print(f"\n>>> Model: {name}")
        print(f"    Accuracy:     {acc:.4f} | 5-Fold CV: {cv_mean:.4f} (+/- {cv_std:.4f})")
        print(f"    Precision:    {prec:.4f}")
        print(f"    Recall:       {rec:.4f}")
        print(f"    F1-Score:     {f1:.4f}")
        print(f"    ROC-AUC:      {roc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=["Rejected (0)", "Approved (1)"]))

    results_df = pd.DataFrame(results)
    print("\n" + "="*70)
    print("                      COMPARATIVE SUMMARY")
    print("="*70)
    print(results_df.to_string(index=False))

    # Identify Best Model
    best_model_idx = results_df["F1-Score"].idxmax()
    best_model_name = results_df.loc[best_model_idx, "Model"]
    best_model = trained_models[best_model_name]

    best_model_path = os.path.join(models_dir, "best_model.pkl")
    joblib.dump(best_model, best_model_path)
    print(f"\n[WINNER] Best Model: {best_model_name}")
    print(f"[SUCCESS] Best model artifact saved to {best_model_path}")

    # Plot Model Performance Comparison Bar Chart
    plt.figure(figsize=(10, 5))
    metrics_melted = results_df.melt(
        id_vars=["Model"],
        value_vars=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
        var_name="Metric",
        value_name="Score"
    )
    sns.barplot(data=metrics_melted, x="Metric", y="Score", hue="Model", palette="viridis")
    plt.title("Classification Algorithm Performance Comparison", fontsize=14, fontweight="bold", pad=15)
    plt.ylim(0.70, 1.0)
    plt.legend(title="Algorithm", frameon=True, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "model_comparison_bar.png"))
    plt.close("all")
    print("  -> Saved model_comparison_bar.png")

    # Plot Confusion Matrix for Best Model
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, best_model.predict(X_test))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted Rejected", "Predicted Approved"],
        yticklabels=["Actual Rejected", "Actual Approved"],
        cbar=False
    )
    plt.title(f"Confusion Matrix ({best_model_name})", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "confusion_matrix_best.png"))
    plt.close("all")
    print("  -> Saved confusion_matrix_best.png")

    # Plot Feature Importance
    if hasattr(best_model, "feature_importances_"):
        plt.figure(figsize=(10, 6))
        importances = best_model.feature_importances_
        fi_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        sns.barplot(data=fi_df.head(10), x="Importance", y="Feature", palette="mako")
        plt.title(f"Top 10 Feature Importances ({best_model_name})", fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("Relative Importance Score", fontsize=12, fontweight="bold")
        plt.ylabel("Feature", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "feature_importance.png"))
        plt.close("all")
        print("  -> Saved feature_importance.png")

    print("\n[SUCCESS] Model training and evaluation completed successfully!")
    return results_df


if __name__ == "__main__":
    train_and_evaluate_models()
