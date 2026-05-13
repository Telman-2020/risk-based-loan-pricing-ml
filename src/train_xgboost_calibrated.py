from pathlib import Path

import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBClassifier


MODEL_DATA_PATH = Path("data/processed/freddie_2023_model_dataset.csv")
REPORTS_DIR = Path("reports")

TARGET = "bad_loan"

DROP_COLUMNS = [
    "loan_sequence_number",
    "source_file",
    "max_delinquency_status",
    "first_reporting_period",
    "last_reporting_period",
    "observation_months",
    "pre_harp_loan_sequence_number",
    "harp_indicator",
]

NUMERIC_CANDIDATES = [
    "credit_score",
    "mortgage_insurance_percentage",
    "number_of_units",
    "original_cltv",
    "original_dti_ratio",
    "original_upb",
    "original_ltv",
    "original_interest_rate",
    "original_loan_term",
    "number_of_borrowers",
]

CATEGORICAL_CANDIDATES = [
    "first_time_homebuyer_flag",
    "occupancy_status",
    "channel",
    "prepayment_penalty_mortgage_flag",
    "amortization_type",
    "property_state",
    "property_type",
    "loan_purpose",
    "super_conforming_flag",
    "program_indicator",
    "property_valuation_method",
    "interest_only_indicator",
    "mi_cancellation_indicator",
    "msa",
    "postal_code",
    "seller_name",
    "servicer_name",
]


def remove_collinear_numeric_features(df, numeric_features, threshold=0.60):
    numeric_df = df[numeric_features].copy()

    for col in numeric_features:
        numeric_df[col] = pd.to_numeric(numeric_df[col], errors="coerce")

    corr = numeric_df.corr(method="pearson").abs()

    features_to_drop = set()
    columns = corr.columns.tolist()

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            feature_1 = columns[i]
            feature_2 = columns[j]
            correlation = corr.loc[feature_1, feature_2]

            if pd.notna(correlation) and correlation > threshold:
                features_to_drop.add(feature_2)

    return [col for col in numeric_features if col not in features_to_drop]


def clean_categorical_features(df, categorical_features, max_unique_values=100):
    selected_features = []

    for col in categorical_features:
        nunique = df[col].fillna("Missing").astype(str).nunique()

        if nunique <= 1:
            continue

        if nunique > max_unique_values:
            continue

        selected_features.append(col)

    return selected_features


def undersample_training_data(X_train, y_train, negative_to_positive_ratio=3):
    train_df = X_train.copy()
    train_df[TARGET] = y_train.values

    positive_df = train_df[train_df[TARGET] == 1]
    negative_df = train_df[train_df[TARGET] == 0]

    n_positive = len(positive_df)
    n_negative_keep = min(len(negative_df), n_positive * negative_to_positive_ratio)

    negative_sample = negative_df.sample(
        n=n_negative_keep,
        random_state=42,
    )

    balanced_df = pd.concat([positive_df, negative_sample], axis=0)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    y_balanced = balanced_df[TARGET].astype(int)
    X_balanced = balanced_df.drop(columns=[TARGET])

    return X_balanced, y_balanced


def build_preprocessor(numeric_features, categorical_features):
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading model dataset...")
    df = pd.read_csv(MODEL_DATA_PATH)

    y = df[TARGET].astype(int)

    available_numeric = [
        col for col in NUMERIC_CANDIDATES
        if col in df.columns and col not in DROP_COLUMNS
    ]

    available_categorical = [
        col for col in CATEGORICAL_CANDIDATES
        if col in df.columns and col not in DROP_COLUMNS
    ]

    selected_numeric = remove_collinear_numeric_features(
        df,
        available_numeric,
        threshold=0.60,
    )

    selected_categorical = clean_categorical_features(
        df,
        available_categorical,
        max_unique_values=100,
    )

    selected_features = selected_numeric + selected_categorical

    print("\nSelected numeric features:")
    print(selected_numeric)

    print("\nSelected categorical features:")
    print(selected_categorical)

    X = df[selected_features].copy()

    for col in selected_numeric:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Train / calibration / test split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    X_train, X_calibration, y_train, y_calibration = train_test_split(
        X_temp,
        y_temp,
        test_size=0.25,
        random_state=42,
        stratify=y_temp,
    )

    print("\nTrain distribution before undersampling:")
    print(y_train.value_counts())
    print(y_train.value_counts(normalize=True).round(4))

    X_train_balanced, y_train_balanced = undersample_training_data(
        X_train,
        y_train,
        negative_to_positive_ratio=3,
    )

    print("\nTrain distribution after undersampling:")
    print(y_train_balanced.value_counts())
    print(y_train_balanced.value_counts(normalize=True).round(4))

    preprocessor = build_preprocessor(
        selected_numeric,
        selected_categorical,
    )

    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.80,
        colsample_bytree=0.80,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    base_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", xgb_model),
        ]
    )

    print("\nTraining XGBoost model on undersampled data...")
    base_pipeline.fit(X_train_balanced, y_train_balanced)

    print("\nCalibrating XGBoost probabilities on untouched calibration data...")
    calibrated_model = CalibratedClassifierCV(
        estimator=FrozenEstimator(base_pipeline),
        method="sigmoid",
    )

    calibrated_model.fit(X_calibration, y_calibration)

    raw_score = base_pipeline.predict_proba(X_test)[:, 1]
    calibrated_pd = calibrated_model.predict_proba(X_test)[:, 1]

    raw_roc_auc = roc_auc_score(y_test, raw_score)
    raw_pr_auc = average_precision_score(y_test, raw_score)

    calibrated_roc_auc = roc_auc_score(y_test, calibrated_pd)
    calibrated_pr_auc = average_precision_score(y_test, calibrated_pd)

    print("\nRaw XGBoost results:")
    print(f"ROC-AUC: {raw_roc_auc:.4f}")
    print(f"PR-AUC: {raw_pr_auc:.4f}")

    print("\nCalibrated XGBoost results:")
    print(f"ROC-AUC: {calibrated_roc_auc:.4f}")
    print(f"PR-AUC: {calibrated_pr_auc:.4f}")

    y_pred = (calibrated_pd >= 0.03).astype(int)

    print("\nClassification report using calibrated PD threshold 0.03:")
    print(classification_report(y_test, y_pred, digits=4))

    predictions_df = X_test.copy()
    predictions_df["actual_bad_loan"] = y_test.values
    predictions_df["raw_xgboost_score"] = raw_score
    predictions_df["calibrated_pd"] = calibrated_pd

    output_path = REPORTS_DIR / "xgboost_test_predictions.csv"
    predictions_df.to_csv(output_path, index=False)

    print("\nSaved XGBoost calibrated predictions:")
    print(output_path)


if __name__ == "__main__":
    main()