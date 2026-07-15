# Hospital Readmission Risk Predictor

A phased MLOps build predicting whether a patient will be readmitted to the hospital within 30 days of discharge, using the UCI Diabetes 130-US Hospitals dataset.

## Problem

Hospitals in the U.S. are financially penalized under CMS's Hospital Readmissions Reduction Program when their 30-day readmission rates are too high. This project builds a model — and the system around it — to flag high-risk patients before discharge so care teams can intervene early.

## Dataset

[UCI Diabetes 130-US Hospitals for Years 1999–2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)

- ~101,766 patient encounters across 130 U.S. hospitals (1999–2008)
- 50+ features per encounter (demographics, diagnoses, medications, prior utilization)
- Licensed CC BY 4.0, downloadable directly as CSV, no login required

Citation: Clore, J., et al. "Diabetes 130-US Hospitals for Years 1999-2008." UCI Machine Learning Repository, 2014. https://doi.org/10.24432/C5230J

## Approach

One dataset, three modeling tasks:

| Type | Model | Purpose |
|---|---|---|
| Supervised (required) | Logistic Regression → Random Forest / XGBoost | Predict binary 30-day readmission label |
| Unsupervised #1 | K-Means / hierarchical clustering | Group patients into risk profiles/segments |
| Unsupervised #2 | PCA (or t-SNE) | Reduce 50+ features to a handful of components; visualize separation, optionally feed engineered components into the supervised model |

The target column is collapsed from the dataset's native three classes (`<30`, `>30`, `NO`) into a binary label (`<30` = 1, everything else = 0), since the project is specifically about the 30-day window.

## Roadmap

**Phase 1 — Required build (Weeks 1–3)**
- [ ] Clean and engineer features
- [ ] Run clustering and PCA as exploratory unsupervised analysis
- [ ] Train supervised model as a callable Python script (`main()`, not notebook-only)
- [ ] Log training runs to MLflow (params, metrics, model artifact)
- [ ] Manually promote best run to "Production" in the MLflow Model Registry
- [ ] Build a FastAPI endpoint that serves predictions from whatever model is tagged "Production"

Phase 1 alone is a complete, defensible capstone: train → track → serve, with class imbalance handling and supporting unsupervised analysis.

**Phase 2 — Stretch goal (Weeks 4–6)**
- [ ] Wrap the Phase 1 training script in an Airflow DAG (retrain task)
- [ ] Add an evaluation task comparing the new model's metrics against current Production
- [ ] Add a conditional promote task — only flips Production if the new model wins
- [ ] FastAPI needs no changes — it already reads whatever's tagged Production

## Tech Stack

- **Language:** Python
- **Modeling:** pandas, scikit-learn, XGBoost
- **Experiment tracking:** MLflow
- **Serving:** FastAPI
- **Orchestration (Phase 2):** Airflow, docker-compose
- **Dev environment:** JupyterLab

## Repository Structure

```
.
├── data/               # raw and processed data (not committed — see .gitignore)
├── notebooks/          # exploratory analysis, clustering, PCA
├── src/
│   ├── features.py     # feature engineering / preprocessing
│   ├── train.py        # supervised model training (callable main())
│   └── serve.py         # FastAPI app
├── mlruns/             # MLflow tracking (not committed)
├── dags/                # Airflow DAGs (Phase 2)
├── requirements.txt
└── README.md
```

## Getting Started

```bash
git clone https://github.com/[your-username]/hospital-readmission-risk-predictor.git
cd hospital-readmission-risk-predictor
pip install -r requirements.txt
```

Download the dataset from the [UCI link above](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) and place the CSV in `data/`.

## Status

🚧 In progress — Phase 1 build underway.

## Author

Angeline Setiawan
