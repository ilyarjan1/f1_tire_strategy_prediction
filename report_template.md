# KNN Project Report – F1 Strategy Dataset (Pit Stop Prediction)

## 1) Introduction
- **Problem statement:** Predict the pit-stop related class label from race/strategy measurements using KNN.
- **Dataset description:**
  - Number of rows:
  - Number of features:
  - Measurement types (numeric / categorical / mixed):
  - Number of classes and class labels:

## 2) Data Discovery & Visualization
Include and discuss:
- Distribution plots (`figures/numeric_distributions.png`)
- Correlation heatmap (`figures/correlation_heatmap.png`)
- PCA scatter for cluster intuition (`figures/pca_scatter.png`)

Discussion prompts:
- Which features appear related?
- Any visible grouping/clustering by class?
- Any skewed or outlier-heavy features?

## 3) Data Preparation
Describe:
- Missing-value handling (median for numeric, most frequent for categorical).
- Outlier handling (IQR clipping for numeric features).
- Encoding of categorical variables (one-hot encoding).
- Standardization of numeric variables (required for KNN distance fairness).

## 4) Data Partitioning
- Train-test split ratio: 80/20
- Stratified by target classes: Yes
- Random seed: 42

## 5) Different Values of K
Chosen K values: **3, 7, 15**.

Reasoning (sample):
- **K=3**: low bias, captures local structure, but can be noisy.
- **K=7**: balanced variance/bias midpoint.
- **K=15**: smoother boundary, less sensitivity to noise, may underfit.

## 6) Training Phase
Discuss KNN training conceptually:
1. Compute distances to all training points.
2. Select K nearest neighbors.
3. Predict class by majority vote.

Use training accuracy from `outputs/model_summary.csv`.

## 7) Testing Phase
Compare training vs test accuracy for each K and discuss:
- overfitting signs (high train, lower test),
- underfitting signs (both low),
- best generalization.

## 8) Evaluation Phase
For each K, include:
- Confusion matrix (`outputs/confusion_matrix_k*.csv`)
- Precision and Recall (`outputs/classification_report_k*.csv`)
- Misclassification error = 1 - test accuracy

Discuss which classes are hardest to classify and why.

## 9) Best Model
Select the best model based on:
1. Highest test accuracy (primary),
2. Higher macro-F1 as tie-breaker.

Suggested improvements:
- Hyperparameter tuning with cross-validation,
- Feature engineering,
- Class balancing methods if imbalance exists,
- Alternative distance metrics.

## 10) Conclusion
Summarize final model performance, practical interpretation, and future work.
