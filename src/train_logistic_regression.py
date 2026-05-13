from pathlib import Path

import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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
    dropped_pairs = []

    columns = corr.columns.tolist()

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            feature_1 = columns[i]
            feature_2 = columns[j]
            correlation = corr.loc[feature_1, feature_2]

            if pd.notna(correlation) and correlation > threshold:
                features_to_drop.add(feature_2)
                dropped_pairs.append(
                    {
                        "kept_feature": feature_1,
                        "dropped_feature": feature_2,
                        "absolute_pearson_correlation": round(correlation, 2),
                    }
                )

    selected_numeric_features = [
        col for col in numeric_features if col not in features_to_drop
    ]

    return selected_numeric_features, pd.DataFrame(dropped_pairs)


def clean_categorical_features(df, categorical_features, max_unique_values=100):
    selected_features = []
    dropped_records = []

    for col in categorical_features:
        nunique = df[col].fillna("Missing").astype(str).nunique()

        if nunique <= 1:
            dropped_records.append(
                {
                    "feature": col,
                    "reason": "only_one_class",
                    "unique_values": nunique,
                }
            )
        elif nunique > max_unique_values:
            dropped_records.append(
                {
                    "feature": col,
                    "reason": "high_cardinality_or_redundant",
                    "unique_values": nunique,
                }
            )
        else:
            selected_features.append(col)

    return selected_features, pd.DataFrame(dropped_records)


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

    selected_numeric, dropped_collinear_df = remove_collinear_numeric_features(
        df,
        available_numeric,
        threshold=0.60,
    )

    selected_categorical, dropped_categorical_df = clean_categorical_features(
        df,
        available_categorical,
        max_unique_values=100,
    )

    dropped_collinear_df.to_csv(
        REPORTS_DIR / "dropped_collinear_numeric_features.csv",
        index=False,
    )

    dropped_categorical_df.to_csv(
        REPORTS_DIR / "dropped_categorical_features.csv",
        index=False,
    )

    selected_features = selected_numeric + selected_categorical
    X = df[selected_features].copy()

    for col in selected_numeric:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # First split: hold out final test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Second split: create calibration set from remaining data
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

    print("\nCalibration distribution:")
    print(y_calibration.value_counts())
    print(y_calibration.value_counts(normalize=True).round(4))

    print("\nTest distribution:")
    print(y_test.value_counts())
    print(y_test.value_counts(normalize=True).round(4))

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, selected_numeric),
            ("categorical", categorical_pipeline, selected_categorical),
        ]
    )
########################################
    candidate_models = {
    "logistic_regression_l1": LogisticRegression(
        l1_ratio=1.0,
        solver="liblinear",
        C=1.0,
        max_iter=300,
        random_state=42,
    ),
    "logistic_regression_l2": LogisticRegression(
        l1_ratio=0.0,
        solver="liblinear",
        C=1.0,
        max_iter=300,
        random_state=42,
    ),
}

    model_results = []
    trained_pipelines = {}

    for model_name, model in candidate_models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        print(f"\nTraining {model_name} on undersampled data...")
        pipeline.fit(X_train_balanced, y_train_balanced)

        validation_score = pipeline.predict_proba(X_calibration)[:, 1]

        validation_roc_auc = roc_auc_score(y_calibration, validation_score)
        validation_pr_auc = average_precision_score(y_calibration, validation_score)

        model_results.append(
            {
                "model_name": model_name,
                "validation_roc_auc": round(validation_roc_auc, 4),
                "validation_pr_auc": round(validation_pr_auc, 4),
            }
        )

        trained_pipelines[model_name] = pipeline

    model_results_df = pd.DataFrame(model_results).sort_values(
        "validation_pr_auc",
        ascending=False,
    )

    model_results_df.to_csv(
        REPORTS_DIR / "logistic_regression_l1_l2_comparison.csv",
        index=False,
    )

    print("\nL1 vs L2 validation comparison:")
    print(model_results_df)

    best_model_name = model_results_df.iloc[0]["model_name"]
    base_pipeline = trained_pipelines[best_model_name]

    print(f"\nSelected best model for calibration: {best_model_name}")

    ####################################

    print("\nTraining Logistic Regression model on undersampled data...")
    base_pipeline.fit(X_train_balanced, y_train_balanced)

    print("\nCalibrating probabilities on untouched calibration data...")
    
    
    calibrated_model = CalibratedClassifierCV(
    estimator=FrozenEstimator(base_pipeline),
    method="sigmoid",
    )

    calibrated_model.fit(X_calibration, y_calibration)

    y_raw_score = base_pipeline.predict_proba(X_test)[:, 1]
    y_calibrated_score = calibrated_model.predict_proba(X_test)[:, 1]

    raw_roc_auc = roc_auc_score(y_test, y_raw_score)
    raw_pr_auc = average_precision_score(y_test, y_raw_score)

    calibrated_roc_auc = roc_auc_score(y_test, y_calibrated_score)
    calibrated_pr_auc = average_precision_score(y_test, y_calibrated_score)

    print("\nRaw Logistic Regression results:")
    print(f"ROC-AUC: {raw_roc_auc:.4f}")
    print(f"PR-AUC: {raw_pr_auc:.4f}")

    print("\nCalibrated Logistic Regression results:")
    print(f"ROC-AUC: {calibrated_roc_auc:.4f}")
    print(f"PR-AUC: {calibrated_pr_auc:.4f}")

    y_pred = (y_calibrated_score >= 0.05).astype(int)

    print("\nClassification report using calibrated PD threshold 0.05:")
    print(classification_report(y_test, y_pred, digits=4))

    predictions_df = X_test.copy()
    predictions_df["actual_bad_loan"] = y_test.values
    predictions_df["raw_risk_score"] = y_raw_score
    predictions_df["calibrated_pd"] = y_calibrated_score

    predictions_output_path = REPORTS_DIR / "logistic_regression_test_predictions.csv"
    predictions_df.to_csv(predictions_output_path, index=False)

    print("\nSaved calibrated test predictions:")
    print(predictions_output_path)


if __name__ == "__main__":
    main()