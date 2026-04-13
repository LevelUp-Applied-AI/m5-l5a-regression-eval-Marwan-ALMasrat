"""
Module 5 Week A — Lab: Regression & Evaluation
BEFORE Feature Engineering
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay,
                             accuracy_score, precision_score,
                             recall_score, f1_score,
                             mean_absolute_error, r2_score)


def load_data(filepath="data/telecom_churn.csv"):
    df = pd.read_csv(filepath)
    if "customer_id" in df.columns:
        df = df.drop(columns=["customer_id"])
        print("Dropped 'customer_id' column")
    print("\n=== Basic EDA ===")
    print(f"Shape: {df.shape}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nChurn distribution:\n{df['churned'].value_counts()}")
    print(f"Churn rate: {df['churned'].mean():.2%}")
    return df


def split_data(df, target_col, test_size=0.2, random_state=42):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    stratify = y if y.nunique() <= 10 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    print(f"\n=== Split: target='{target_col}' ===")
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    if stratify is not None:
        print(f"Train churn rate: {y_train.mean():.2%}")
        print(f"Test  churn rate: {y_test.mean():.2%}")
    return X_train, X_test, y_train, y_test


def build_logistic_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            random_state=42, max_iter=1000, class_weight="balanced"
        ))
    ])


def build_ridge_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", Ridge(alpha=1.0))
    ])


def build_lasso_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", Lasso(alpha=0.1))
    ])


def evaluate_classifier(pipeline, X_train, X_test, y_train, y_test):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    print("=== Confusion Matrix ===")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    print(cm)
    return {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
    }


def evaluate_regressor(pipeline, X_train, X_test, y_train, y_test):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    return {
        "mae": mean_absolute_error(y_test, y_pred),
        "r2":  r2_score(y_test, y_pred),
    }


def run_cross_validation(pipeline, X_train, y_train, cv=5):
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X_train, y_train,
                             cv=cv_splitter, scoring="accuracy")
    print("\n=== Cross-Validation Scores ===")
    for i, score in enumerate(scores, 1):
        print(f"  Fold {i}: {score:.4f}")
    print(f"  Mean: {scores.mean():.3f} +/- {scores.std():.3f}")
    return scores


def random_forest_feature_importance(X_train, y_train, X_test, y_test):
    print("\n" + "=" * 50)
    print("RANDOM FOREST — Feature Importance")
    print("=" * 50)
    rf = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print("\n=== Random Forest Classification Report ===")
    print(classification_report(y_test, y_pred))
    importances = pd.Series(
        rf.feature_importances_, index=X_train.columns
    ).sort_values(ascending=False)
    print(f"\n{'Feature':<25} {'Importance':>10}")
    print("-" * 37)
    for feat, imp in importances.items():
        bar = "█" * int(imp * 100)
        print(f"{feat:<25} {imp:>10.4f}  {bar}")
    return importances


if __name__ == "__main__":
    df = load_data()
    if df is not None:
        print(f"\nLoaded {len(df)} rows, {df.shape[1]} columns")

        numeric_features = ["tenure", "monthly_charges", "total_charges",
                            "num_support_calls", "senior_citizen",
                            "has_partner", "has_dependents"]

        # Classification
        print("\n" + "=" * 50)
        print("CLASSIFICATION: Logistic Regression → churned")
        print("=" * 50)
        df_cls = df[numeric_features + ["churned"]].dropna()
        split = split_data(df_cls, "churned")
        if split:
            X_train, X_test, y_train, y_test = split
            pipe = build_logistic_pipeline()
            metrics = evaluate_classifier(pipe, X_train, X_test, y_train, y_test)
            print(f"\nLogistic Regression metrics: {metrics}")
            scores = run_cross_validation(pipe, X_train, y_train)
            print(f"CV accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")

        # Random Forest
        if split:
            random_forest_feature_importance(X_train, y_train, X_test, y_test)

        # Ridge Regression
        print("\n" + "=" * 50)
        print("REGRESSION (Ridge): predict monthly_charges")
        print("=" * 50)
        df_reg = df[["tenure", "total_charges", "num_support_calls",
                     "senior_citizen", "has_partner", "has_dependents",
                     "monthly_charges"]].dropna()
        split_reg = split_data(df_reg, "monthly_charges")
        if split_reg:
            X_tr, X_te, y_tr, y_te = split_reg
            ridge_pipe = build_ridge_pipeline()
            reg_metrics = evaluate_regressor(ridge_pipe, X_tr, X_te, y_tr, y_te)
            print(f"\nRidge Regression metrics: {reg_metrics}")

        # Lasso vs Ridge
        print("\n" + "=" * 50)
        print("REGRESSION (Lasso vs Ridge): coefficient comparison")
        print("=" * 50)
        if split_reg:
            lasso_pipe = build_lasso_pipeline()
            lasso_pipe.fit(X_tr, y_tr)
            ridge_pipe.fit(X_tr, y_tr)
            feature_names = X_tr.columns.tolist()
            ridge_coefs = ridge_pipe.named_steps["regressor"].coef_
            lasso_coefs = lasso_pipe.named_steps["regressor"].coef_
            print(f"\n{'Feature':<22} {'Ridge':>10} {'Lasso':>10}")
            print("-" * 44)
            for feat, r, l in zip(feature_names, ridge_coefs, lasso_coefs):
                zeroed = " <- zeroed" if abs(l) < 1e-6 else ""
                print(f"{feat:<22} {r:>10.4f} {l:>10.4f}{zeroed}")
"""
TASK 7 — SUMMARY OF FINDINGS
==============================

1. Most Important Features for Predicting Churn:

   Based on Ridge/Lasso coefficients and Random Forest feature importances:

   BEFORE feature engineering:
   - tenure (RF: 0.24, Ridge: -26.17): strongest predictor — customers
     with longer tenure are far less likely to churn.
   - total_charges (RF: 0.29, Ridge: +35.30): highly correlated with
     churn risk, reflecting accumulated billing pressure over time.
   - monthly_charges (RF: 0.28): current bill amount is a direct churn
     driver, independent of how long the customer has stayed.
   - num_support_calls (RF: 0.10): customers who contact support
     frequently show higher churn likelihood, suggesting dissatisfaction.
   - Demographic flags (senior_citizen, has_partner, has_dependents)
     show weak influence (~0.03 each) and were nearly zeroed by Lasso,
     confirming they add little predictive value.

   AFTER feature engineering (4 new features added):
   - avg_charge_per_month (RF: 0.18) became the top feature, surpassing
     total_charges and monthly_charges individually — the actual average
     payment per month captures churn signal more efficiently than either
     raw column alone.
   - charge_gap (RF: 0.17): the difference between expected and actual
     total payments proved meaningful — customers with large gaps may
     have received discounts that masked underlying dissatisfaction.
   - Lasso zeroed out num_support_calls, senior_citizen, has_dependents,
     is_new_customer, and high_support_user in the regression task,
     confirming these do not linearly predict monthly_charges.


2. Logistic Regression Performance:

   BEFORE feature engineering:
   - Accuracy: 63% | Precision: 22% | Recall: 51% | F1: 31%
   - CV Mean: 0.607 +/- 0.019

   AFTER feature engineering:
   - Accuracy: 68% | Precision: 24% | Recall: 45% | F1: 31%
   - CV Mean: 0.637 +/- 0.017

   Accuracy is misleading on this imbalanced dataset (~16% churn).
   Feature engineering improved accuracy (+5%) and CV mean (+3%),
   but recall dropped from 51% to 45% while F1 remained unchanged
   at 0.31 — meaning the overall precision/recall balance did not improve.

   Recall is more critical here: missing a churner (false negative)
   means losing a customer with no intervention opportunity, which is
   far more costly than a false alarm (false positive) that only wastes
   a retention offer. The low std in CV (0.019 → 0.017) confirms the
   model is stable and not overfitting, but overall performance remains
   limited regardless of the feature set used.

   Ridge regression improved dramatically after feature engineering:
   R² jumped from 0.71 to 0.99 and MAE dropped from 10.61 to 1.86,
   driven by avg_charge_per_month and charge_gap which are near-linear
   transformations of monthly_charges itself.


3. Recommended Next Steps:

   - Apply feature engineering: the 4 engineered features
     (avg_charge_per_month, charge_gap, is_new_customer,
     high_support_user) redistributed importance more evenly across
     features and improved CV accuracy, but recall needs further work.

   - Add categorical features (contract_type, internet_service,
     payment_method) via OneHotEncoder inside a ColumnTransformer —
     contract type in particular is a well-known churn predictor in
     telecom as month-to-month customers churn at much higher rates.

   - Lower the classification threshold below 0.5 using predict_proba
     to recover the recall lost after feature engineering, which better
     aligns with the business goal of catching churners early.

   - Try Random Forest with class_weight="balanced" as the primary
     classifier. The current RF (recall=0.02) ignores the minority
     class without balancing — once fixed, its ability to capture
     non-linear interactions should outperform logistic regression.

   - Compare against a DummyClassifier baseline to confirm the model
     adds real value beyond simply predicting the majority class.
"""