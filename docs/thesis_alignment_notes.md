# Thesis Alignment Notes

This codebase corresponds to experiments described in the PhD thesis "A Deep Reinforcement Learning Framework with Automatic Curriculum Generation for Cryptocurrency Price Prediction".

- **notebooks/PhD.ipynb** reproduces the end-to-end pipeline.
- **src/curriculum_bandit.py** implements UCB and Thompson Sampling teachers for ACG.
- **src/model_training.py** trains the student (KAN, LSTM, Transformer baselines).
- **src/data_preprocessing.py** includes leak-safe splits and feature engineering.
- **src/evaluation_metrics.py** computes MAE, RMSE, DA and supports Wilcoxon tests.
