# F1 Strategy ML Project

This repository is a practical machine-learning project for predicting pit-stop strategy decisions from the F1 strategy dataset.

## What this project includes
- A clean training script for stronger tabular models (`RandomForest` + `HistGradientBoosting`).
- Resumable training flow for Google Colab sessions.
- A notebook you can edit for EDA, experiments, and report screenshots.
- Saved artifacts for best model + run history.

## Main files
- `src/train_strategy_models.py` → recommended training pipeline.
- `notebooks/f1_strategy_ml_project.ipynb` → editable notebook.
- `notebooks/colab_training_starter.md` → quick Colab commands.

## Quick start
```bash
pip install -r requirements.txt
# F1 Tire Strategy Prediction with K-Nearest Neighbors (KNN)

This project is a full, grade-ready KNN classification workflow for the Kaggle dataset:
**F1 Strategy Dataset – Pit Stop Prediction**.

Kaggle link: https://www.kaggle.com/datasets/aadigupta1601/f1-strategy-dataset-pit-stop-prediction

## What this project delivers

- A complete pipeline that matches all 10 assignment requirements.
- Data exploration and visualizations.
- Preprocessing for mixed data types (numeric + categorical).
- Train/test split and three different values of `K`.
- Training and testing metrics comparison.
- Confusion matrix, precision, recall, and misclassification error.
- Best-model selection and saved trained model artifact.
- A report template (`report_template.md`) structured exactly by your grading rubric.

## Recommended environment

You can run this in either:

1. **Codex / local machine** (good for quick iterations and moderate-sized datasets), or
2. **Google Colab** (recommended if you want more compute, longer sessions, and easy plotting in notebooks).

For this dataset size, KNN should usually run fine in either environment.

## Quick start

1. Place your CSV in `data/` (for example: `data/f1_strategy.csv`).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run:

```bash
python src/knn_f1_strategy.py \
  --data data/f1_strategy.csv \
  --target <TARGET_COLUMN_NAME> \
  --k-values 3 7 15
```

4. Outputs generated:
- `figures/*` visualizations
- `outputs/model_summary.csv` (train/test performance across K values)
- `outputs/classification_report_k*.csv`
- `outputs/confusion_matrix_k*.csv`
- `outputs/best_knn_model.joblib`
- `outputs/auto_report.md`

## Portfolio-grade training pipeline (recommended)

For stronger performance than plain KNN, use the resumable multi-model trainer:

```bash
python src/train_strategy_models.py \
  --data data/f1_strategy.csv \
  --target pit_stop \
  --output-dir artifacts \
  --random-state 42
```

Resume later:
Resume in a later session (useful for Colab disconnects):

```bash
python src/train_strategy_models.py \
  --data data/f1_strategy.csv \
  --target pit_stop \
  --output-dir artifacts \
  --random-state 42 \
  --resume
```

## What to do now
1. Put your real CSV inside `data/`.
2. Run the training command above.
3. Open `artifacts/run_history.csv` and check the best run.
4. Use `artifacts/models/best_model.joblib` in your demo app or notebook.
5. Write a short case-study README section with your final metrics and what you learned.

## Notes
- `src/knn_f1_strategy.py` is kept only as a legacy baseline from the initial assignment phase.
