# Colab Starter Commands

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
!pip install -q pandas numpy scikit-learn joblib
```

```bash
!python src/train_strategy_models.py \
  --data data/f1_strategy.csv \
  --target pit_stop \
  --output-dir artifacts \
  --random-state 42
```

Resume later session:

```bash
!python src/train_strategy_models.py \
  --data data/f1_strategy.csv \
  --target pit_stop \
  --output-dir artifacts \
  --random-state 42 \
  --resume
```
