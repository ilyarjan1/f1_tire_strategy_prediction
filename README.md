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

## How this maps to your assignment

Each rubric item (Introduction, Exploration, Preprocessing, Split, K choices, Training, Testing, Evaluation, Best model, Conclusion) is scaffolded in `report_template.md` and auto-supported by the outputs produced by the script.
