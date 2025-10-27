<<<<<<< HEAD
# deep-rl-crypto-acg
Deep Reinforcement Learning Framework with Automatic Curriculum Generation for Cryptocurrency Price Prediction — PhD Work (JKUAT)
=======
# Deep Reinforcement Learning Framework with Automatic Curriculum Generation for Cryptocurrency Price Prediction

This repository contains the official implementation of the PhD research by **David Gichuiri Kibaara (JKUAT, 2021–2025)**.

It implements a **bandit-driven Teacher–Student Curriculum Learning (TSCL)** framework for **cryptocurrency time-series forecasting** with **automatic curriculum generation (ACG)** using multi-armed bandits (UCB, Thompson Sampling) and student models (KAN and deep learning baselines).

---

## 📁 Repository Structure

```
.
├── README.md
├── LICENSE
├── .gitignore
├── environment.yml
├── requirements.txt
├── open_science_manifest.txt
├── data/
│   └── crypto_prices_sample.csv
├── docs/
│   └── thesis_alignment_notes.md
├── notebooks/
│   └── PhD.ipynb
├── results/
│   ├── figures/
│   └── tables/
└── src/
    ├── curriculum_bandit.py
    ├── data_preprocessing.py
    ├── evaluation_metrics.py
    └── model_training.py
```

---

## ⚙️ Setup

```bash
conda env create -f environment.yml
conda activate drl-crypto
```

Alternatively (pip only):

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## ▶️ Quickstart

1. Launch Jupyter and open the notebook:
   ```bash
   jupyter notebook notebooks/PhD.ipynb
   ```
2. Or run the pipeline modules:
   ```bash
   python -m src.data_preprocessing
   python -m src.model_training
   ```

---

## 🧠 Citing

Kibaara, D. G. (2025). *A Deep Reinforcement Learning Framework with Automatic Curriculum Generation for Cryptocurrency Price Prediction.* JKUAT PhD Thesis.

---

## 🔒 License

MIT License (see `LICENSE`).
>>>>>>> eceba9f (Initial public release – PhD framework)
