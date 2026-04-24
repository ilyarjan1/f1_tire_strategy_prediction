"""End-to-end KNN classification pipeline for F1 pit-stop strategy prediction."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass
class ModelResult:
    k: int
    train_accuracy: float
    test_accuracy: float
    macro_f1: float
    misclassification_error: float


class IQRClipper(BaseEstimator, TransformerMixin):
    """Clip numeric values by IQR bounds learned from the training set."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        q1 = np.nanpercentile(X, 25, axis=0)
        q3 = np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - 1.5 * iqr
        self.upper_ = q3 + 1.5 * iqr
        return self

    def transform(self, X: pd.DataFrame):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate KNN for F1 strategy prediction.")
    parser.add_argument("--data", required=True, help="Path to input CSV file.")
    parser.add_argument("--target", required=True, help="Target column (class label) in the CSV.")
    parser.add_argument(
        "--k-values",
        nargs=3,
        type=int,
        default=[3, 7, 15],
        help="Exactly three K values to evaluate.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def ensure_dirs() -> None:
    Path("figures").mkdir(parents=True, exist_ok=True)
    Path("outputs").mkdir(parents=True, exist_ok=True)


def load_data(csv_path: str, target: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. Available columns: {list(df.columns)}")
    return df


def plot_discovery_graphs(df: pd.DataFrame, target: str) -> None:
    numeric_df = df.select_dtypes(include=[np.number])

    if not numeric_df.empty:
        axes = numeric_df.hist(figsize=(14, 10), bins=25)
        plt.suptitle("Numeric Feature Distributions", y=1.02)
        plt.tight_layout()
        plt.savefig("figures/numeric_distributions.png", dpi=150)
        plt.close()

        corr = numeric_df.corr(numeric_only=True)
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, cmap="coolwarm", center=0)
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig("figures/correlation_heatmap.png", dpi=150)
        plt.close()

    # PCA view using one-hot encoding for quick cluster intuition.
    X = df.drop(columns=[target])
    y = df[target].astype(str)
    X_encoded = pd.get_dummies(X, drop_first=False)
    X_filled = X_encoded.fillna(X_encoded.median(numeric_only=True))

    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(X_filled)
    pca_df = pd.DataFrame({"PC1": components[:, 0], "PC2": components[:, 1], "target": y})

    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="target", alpha=0.7)
    plt.title("PCA Scatter Plot (2D)")
    plt.tight_layout()
    plt.savefig("figures/pca_scatter.png", dpi=150)
    plt.close()


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("iqr_clipper", IQRClipper()),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def train_and_evaluate(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    k_values: Iterable[int],
) -> tuple[list[ModelResult], dict[int, Pipeline]]:
    results: list[ModelResult] = []
    models: dict[int, Pipeline] = {}

    for k in k_values:
        preprocessor = make_preprocessor(X_train)
        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("knn", KNeighborsClassifier(n_neighbors=k)),
            ]
        )

        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        macro_f1 = f1_score(y_test, y_test_pred, average="macro")

        results.append(
            ModelResult(
                k=k,
                train_accuracy=train_acc,
                test_accuracy=test_acc,
                macro_f1=macro_f1,
                misclassification_error=1 - test_acc,
            )
        )
        models[k] = model

        report = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)
        pd.DataFrame(report).transpose().to_csv(f"outputs/classification_report_k{k}.csv", index=True)

        cm = confusion_matrix(y_test, y_test_pred)
        labels = np.unique(np.concatenate([y_test.astype(str), y_test_pred.astype(str)]))
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        cm_df.to_csv(f"outputs/confusion_matrix_k{k}.csv", index=True)

    return results, models


def save_summary_and_best_model(results: list[ModelResult], models: dict[int, Pipeline]) -> None:
    df_results = pd.DataFrame([r.__dict__ for r in results]).sort_values(by=["test_accuracy", "macro_f1"], ascending=False)
    df_results.to_csv("outputs/model_summary.csv", index=False)

    best_k = int(df_results.iloc[0]["k"])
    best_model = models[best_k]
    joblib.dump(best_model, "outputs/best_knn_model.joblib")

    with open("outputs/auto_report.md", "w", encoding="utf-8") as f:
        f.write("# Auto-generated Model Summary\n\n")
        f.write("## Performance table\n\n")
        f.write(df_results.to_string(index=False))
        f.write("\n\n")
        f.write(f"**Selected best model:** K={best_k} (highest test accuracy, tie-break by macro-F1).\n")


def print_dataset_intro(df: pd.DataFrame, target: str) -> None:
    n_rows, n_cols = df.shape
    feature_count = n_cols - 1
    classes = sorted(df[target].dropna().astype(str).unique().tolist())
    numeric_count = df.drop(columns=[target]).select_dtypes(include=[np.number]).shape[1]
    categorical_count = feature_count - numeric_count

    print("=== INTRODUCTION SNAPSHOT ===")
    print(f"Rows: {n_rows}")
    print(f"Total columns: {n_cols} (features: {feature_count}, target: 1)")
    print(f"Feature types -> numeric: {numeric_count}, categorical: {categorical_count}")
    print(f"Number of classes in '{target}': {len(classes)}")
    print(f"Class labels: {classes}")


def main() -> None:
    args = parse_args()
    ensure_dirs()

    df = load_data(args.data, args.target)
    print_dataset_intro(df, args.target)
    plot_discovery_graphs(df, args.target)

    X = df.drop(columns=[args.target])
    y = df[args.target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    results, models = train_and_evaluate(X_train, X_test, y_train, y_test, args.k_values)
    save_summary_and_best_model(results, models)

    print("\nPipeline complete. Check 'figures/' and 'outputs/' for deliverables.")


if __name__ == "__main__":
    main()
