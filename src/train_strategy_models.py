"""Train multiple tabular models for F1 strategy prediction with resumable experiment tracking.

Colab-friendly features:
- periodic metric/model persistence
- resume from previous sessions
- best-model checkpointing
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier


@dataclass
class RunResult:
    model_name: str
    run_id: str
    accuracy: float
    macro_f1: float
    macro_precision: float
    macro_recall: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--output-dir", default="artifacts")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def build_preprocessor(df: pd.DataFrame, target: str) -> ColumnTransformer:
    X = df.drop(columns=[target])
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])


def get_models(random_state: int) -> dict[str, Any]:
    return {
        "rf_300": RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1),
        "hgb_default": HistGradientBoostingClassifier(random_state=random_state),
        "hgb_deep": HistGradientBoostingClassifier(max_depth=10, learning_rate=0.05, random_state=random_state),
    }


def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    models_dir = out / "models"
    models_dir.mkdir(exist_ok=True)

    history_path = out / "run_history.csv"
    best_path = out / "best_model_meta.json"

    df = pd.read_csv(args.data)
    if args.target not in df.columns:
        raise ValueError(f"Target '{args.target}' missing.")

    X = df.drop(columns=[args.target])
    y = df[args.target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    preprocessor = build_preprocessor(df, args.target)
    candidates = get_models(args.random_state)

    history = pd.read_csv(history_path) if args.resume and history_path.exists() else pd.DataFrame()
    completed = set(history["run_id"].tolist()) if not history.empty else set()

    best_score = -1.0
    best_meta: dict[str, Any] = {}
    if args.resume and best_path.exists():
        best_meta = json.loads(best_path.read_text())
        best_score = float(best_meta.get("macro_f1", -1.0))

    new_rows: list[dict[str, Any]] = []
    for name, model in candidates.items():
        run_id = f"{name}_seed{args.random_state}"
        if run_id in completed:
            print(f"Skipping completed run: {run_id}")
            continue

        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        metrics = evaluate(y_test, preds)

        result = RunResult(model_name=name, run_id=run_id, **metrics)
        row = asdict(result)
        new_rows.append(row)

        run_model_path = models_dir / f"{run_id}.joblib"
        joblib.dump(pipe, run_model_path)

        if row["macro_f1"] > best_score:
            best_score = row["macro_f1"]
            best_meta = {**row, "model_path": str(run_model_path)}
            best_path.write_text(json.dumps(best_meta, indent=2))
            joblib.dump(pipe, models_dir / "best_model.joblib")

        print(f"Done {run_id}: acc={row['accuracy']:.4f}, f1={row['macro_f1']:.4f}")

    final_history = pd.concat([history, pd.DataFrame(new_rows)], ignore_index=True)
    final_history = final_history.sort_values(by=["macro_f1", "accuracy"], ascending=False)
    final_history.to_csv(history_path, index=False)

    print("\nTop runs:")
    print(final_history.head(10).to_string(index=False))
    if best_meta:
        print("\nBest model meta:")
        print(json.dumps(best_meta, indent=2))


if __name__ == "__main__":
    main()
