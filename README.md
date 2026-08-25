# Hospital Readmission Risk Predictor
A phased MLOps build predicting whether a patient will be readmitted to the hospital within 30 days of discharge, using the UCI Diabetes 130-US Hospitals dataset.
 
**Status: Phase 1 complete.** The required build train, track, and serve is finished and fulfills every capstone requirement for this project. See the [Roadmap](#roadmap) and [Status](#status) sections below for details.
 
## Problem
Hospitals in the U.S. are financially penalized under CMS's Hospital Readmissions Reduction Program when their 30-day readmission rates are too high. This project builds a model and the system around it to flag high-risk patients before discharge so care teams can intervene early.
 
## Run the Prediction API Locally
The trained model is served through a FastAPI app that reads whichever model version is currently tagged "Production" in the MLflow registry. Follow these steps in order.
 
1. Open **Anaconda Prompt**.
2. Activate the project environment:
   ```
   conda activate pytorch
   ```
3. Move into the repository folder:
   ```
   cd C:\Users\angie\Documents\GitHub\hospital-readmission-risk-predictor\AngelineSetiawan-hospital-readmission-risk-predictor
   ```
4. Start the FastAPI server:
   ```
   uvicorn serve.main:app --reload --port 8000
   ```
5. Wait for this line in the terminal, which confirms the app has finished loading the Production model from MLflow:
   ```
   INFO:     Application startup complete.
   ```
6. Open the interactive API docs in a browser:
   ```
   http://127.0.0.1:8000/docs
   ```
7. On that page, expand the **POST /predict** endpoint and click **Try it out**. Paste the sample request below into the request body and click **Execute**:
   For a "predicted_high_risk": false. 
   ```json
   {
     "features": {
       "time_in_hospital": 5,
       "num_medications": 12,
       "age": "[70-80)"
     }
   }
   ```

   ```json
   {
     "features": {
       "number_inpatient": 5,
       "age": "[70-80)",
       "number_diagnoses": 9
     }
   }
   ```
   The response returns a `readmission_probability`, along with `predicted_high_risk` (true/false based on the model's tuned decision threshold), the `threshold_used`, and the serving model's `model_version` and `run_id`. Any feature left out of the request is treated as missing and imputed by the model's preprocessing pipeline the same way missing values are handled during training, so a partial payload like the one above still returns a valid prediction.
 
A `GET /health` endpoint is also available at `http://127.0.0.1:8000/health` to confirm the server is up and to see which model version and threshold are currently loaded.
 
## Dataset
[UCI Diabetes 130-US Hospitals for Years 1999-2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
- ~101,766 patient encounters across 130 U.S. hospitals (1999-2008)
- 50+ features per encounter (demographics, diagnoses, medications, prior utilization)
- Licensed CC BY 4.0, downloadable directly as CSV, no login required
 
Citation: Clore, J., et al. "Diabetes 130-US Hospitals for Years 1999-2008." UCI Machine Learning Repository, 2014. https://doi.org/10.24432/C5230J
 
## Approach
One dataset, three modeling tasks:
 
| Type | Model | Purpose |
|---|---|---|
| Supervised (required) | Logistic Regression, tuned via L1/L2 and balanced class weighting | Predict binary 30-day readmission label |
| Unsupervised #1 | K-Means clustering | Group patients into risk profiles/segments |
| Unsupervised #2 | PCA, with t-SNE as a supporting visualization | Reduce 240+ one-hot-encoded features to a handful of components; evaluate whether the components separate the outcome classes |
 
The target column is collapsed from the dataset's native three classes (`<30`, `>30`, `NO`) into a binary label (`<30` = 1, everything else = 0), since the project is specifically about the 30-day window.
 
## Roadmap
**Phase 1 - Required build (complete)**
- [x] Clean and engineer features
- [x] Train a baseline supervised model (regularized logistic regression) with 5-fold cross-validation, test-set evaluation, and assumption diagnostics
- [x] Tune the baseline (regularization strength, penalty type, class weighting) and evaluate tree-based ensembles (Random Forest and XGBoost); compare on PR-AUC
- [x] Select a decision threshold on out-of-fold predictions and complete error analysis before candidate selection
- [x] Run clustering as exploratory unsupervised analysis
- [x] Run PCA (with t-SNE as a supporting check) as exploratory unsupervised analysis
- [x] Convert supervised model training into a callable, reproducible pipeline logged directly from the modeling notebook
- [x] Log training runs to MLflow (params, metrics, model artifact)
- [x] Manually promote the best run to "Production" in the MLflow Model Registry
- [x] Build a FastAPI endpoint that serves predictions from whatever model is tagged "Production"
 
Phase 1 is now complete and covers everything necessary to fulfill the capstone requirement: a documented research question and hypothesis, a leak-free preprocessing pipeline, exploratory unsupervised analysis (PCA and k-means) evaluated on its own merits rather than just mentioned, a baseline model compared against tuned alternatives with a deliberately selected decision threshold, and a live serving layer backed by an MLflow-tracked model registry. Full detail on methods, results, and limitations is in the final report (`reports/S7_Hospital_Readmission_Final_Report.docx.pdf`).
 
**Phase 2 - Stretch goal (not started)**
- [ ] Wrap the Phase 1 training script in an Airflow DAG (retrain task)
- [ ] Add an evaluation task comparing the new model's metrics against current Production
- [ ] Add a conditional promote task, which only flips Production if the new model wins
- [ ] FastAPI needs no changes, since it already reads whatever's tagged Production
 
## Tech Stack
- **Language:** Python
- **Modeling:** pandas, scikit-learn, XGBoost
- **Experiment tracking:** MLflow
- **Serving:** FastAPI, uvicorn
- **Orchestration (Phase 2):** Airflow, docker-compose
- **Dev environment:** JupyterLab, Anaconda
 
## Repository Structure
```
.
├── data/
│   ├── raw/                # diabetic_data.csv, IDS_mapping.csv (original UCI extract)
│   └── processed/          # leak-free train/test splits produced by notebook 02
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing_Feature_Engineering.ipynb
│   ├── 03_Baseline_Models.ipynb
│   ├── 04_Hyperparameter_Tuning_Model_Evaluation_V5.ipynb
│   ├── 05_MLflow_Registry_FastAPI.ipynb
│   ├── outputs/             # saved models, metrics, figures, and tables from every notebook
│   └── mlruns/, mlflow.db   # local MLflow tracking store
├── serve/
│   ├── __init__.py
│   └── main.py              # FastAPI app; regenerated by notebook 05, do not hand-edit
├── reports/                 # phase-by-phase written deliverables, including the final report (S7)
├── mlflow.db                 # MLflow registry backing store used by serve/main.py
└── README.md
```
 
## Getting Started
```bash
git clone https://github.com/AngelineSetiawan/AngelineSetiawan-hospital-readmission-risk-predictor.git
cd AngelineSetiawan-hospital-readmission-risk-predictor
```
The UCI dataset (`diabetic_data.csv`, `IDS_mapping.csv`) is already included under `data/raw/`. If you need a fresh copy, it is available at the [UCI link above](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008).
 
To run the notebooks or the API, activate the `pytorch` conda environment used throughout this project (see [Run the Prediction API Locally](#run-the-prediction-api-locally) above for the exact commands).
 
## Status
Phase 1 build complete. EDA, feature engineering, baseline modeling, hyperparameter tuning, unsupervised analysis (PCA and k-means), MLflow experiment tracking and model registry, and the FastAPI serving layer are all finished, tested, and documented in the reports under `reports/`. The final written deliverable is `reports/S7_Hospital_Readmission_Final_Report.docx.pdf`.
 
Phase 2 (Airflow orchestration for automated retraining) remains a stretch goal and has not been started.
 
## Author
Angeline Setiawan
