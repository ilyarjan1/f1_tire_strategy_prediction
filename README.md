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
python src/train_strategy_models.py \
  --data data/f1_strategy.csv \
  --target pit_stop \
  --output-dir artifacts \
  --random-state 42
```

Resume later:
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
