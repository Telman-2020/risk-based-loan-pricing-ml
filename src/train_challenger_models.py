from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBClassifier


MODEL_DATA_PATH = Path("data/processed/freddie_2023_model_dataset.csv")
REPORTS_DIR = Path("reports")

TARGET = "bad_loan"

DROP_COLUMNS = [
    "loan_sequence_number",
    "source_file",

    # Leakage columns from performance history
    "max_delinquency_status",
    "first_reporting_period",
    "last_reporting_period",
    "observation_months",

    # Mostly missing
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


def remove_collinear_numeric_features(
    df: pd.DataFrame,
    numeric_features: list[str],
    threshold: float = 0.60,
) -> list[str]:
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


def clean_categorical_features(
    df: pd.DataFrame,
    categorical_features: list[str],
    max_unique_values: int = 100,
) -> list[str]:
    selected_features = []

    for col in categorical_features:
        nunique = df[col].fillna("Missing").astype(str).nunique()

        if nunique <= 1:
            continue

        if nunique > max_unique_values:
            continue

        selected_features.append(col)

    return selected_features


def undersample_training_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    negative_to_positive_ratio: int = 3,
) -> tuple[pd.DataFrame, pd.Series]:
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


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
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


def evaluate_model(
    model_name: str,
    pipeline: Pipeline,
    X_train_balanced: pd.DataFrame,
    y_train_balanced: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    print(f"\nTraining {model_name}...")
    pipeline.fit(X_train_balanced, y_train_balanced)

    y_score = pipeline.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_score)
    pr_auc = average_precision_score(y_test, y_score)

    print(f"{model_name} ROC-AUC: {roc_auc:.4f}")
    print(f"{model_name} PR-AUC: {pr_auc:.4f}")

    return {
        "model_name": model_name,
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
    }


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

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("\nOriginal training distribution:")
    print(y_train.value_counts())
    print(y_train.value_counts(normalize=True).round(4))

    X_train_balanced, y_train_balanced = undersample_training_data(
        X_train,
        y_train,
        negative_to_positive_ratio=3,
    )

    print("\nTraining distribution after undersampling:")
    print(y_train_balanced.value_counts())
    print(y_train_balanced.value_counts(normalize=True).round(4))

    models = {
        "xgboost_classifier": XGBClassifier(
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
        ),
        "simple_mlp_classifier": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=0.0001,
            batch_size=512,
            learning_rate_init=0.001,
            max_iter=30,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=42,
        ),
    }

    results = []

    for model_name, model in models.items():
        preprocessor = build_preprocessor(
            selected_numeric,
            selected_categorical,
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        result = evaluate_model(
            model_name=model_name,
            pipeline=pipeline,
            X_train_balanced=X_train_balanced,
            y_train_balanced=y_train_balanced,
            X_test=X_test,
            y_test=y_test,
        )

        results.append(result)

    results_df = pd.DataFrame(results).sort_values(
        "pr_auc",
        ascending=False,
    )

    output_path = REPORTS_DIR / "challenger_model_comparison.csv"
    results_df.to_csv(output_path, index=False)

    print("\nChallenger model comparison:")
    print(results_df)

    print("\nSaved:")
    print(output_path)


if __name__ == "__main__":
    main()