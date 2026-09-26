# Week 17 MLOps - Track A: Data Science MLOps Pipeline

This repository branch (`track-a`) contains a complete classical machine learning pipeline for the **Telco Customer Churn** dataset (~7,043 rows) adhering to production MLOps standards: reproducible environment management with `uv`, experiment tracking & model registry via `MLflow`, real-time serving with `FastAPI`, and drift monitoring using `Evidently AI`.

---

## 1. Environment & Reproducibility (uv)

### Problems Solved by `uv`
- **Deterministic Dependency Pinning**: Python projects frequently suffer from transient sub-dependency drift across OS environments. `uv` generates and locks exact package wheels and versions inside `uv.lock`, eliminating conflicts between `scikit-learn`, `mlflow`, `evidently`, and `pandas`.
- **Fast, Isolated Virtual Environment Setup**: `uv` replaces legacy `pip + venv` workflows with native Rust-backed resolution, ensuring sub-second virtualenv recreation.
- **One-Command Reproducibility**: Any team member or CI/CD runner can clone this repository and spin up an identical runtime environment using a single command:
  ```bash
  uv sync
  ```

---

## 2. Experiment Tracking & Model Registry Strategy (MLflow)

### Experiment Variations & Evaluation
We trained and logged three diverse model architectures across identical 75/25 train-test stratified splits under the MLflow experiment `telco-customer-churn`:
1. **Logistic Regression**: Linear baseline with L2 regularization (`C=0.5`, `max_iter=1000`).
2. **Random Forest Classifier**: Non-linear bagging ensemble (`n_estimators=100`, `max_depth=6`).
3. **Gradient Boosting Classifier**: Sequential tree boosting (`n_estimators=120`, `learning_rate=0.1`, `max_depth=4`).

Because customer churn datasets suffer from severe class imbalance (~26% churn rate), relying solely on accuracy is deceptive (a naive non-churn predictor achieves ~74% accuracy). Hence, the model selection was strictly driven by **F1-Score** (balancing precision and recall) and **ROC-AUC**.

### Run Comparison Table

| Model Family | Hyperparameters | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Decision |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | `C=0.5, max_iter=1000` | **0.8064** | 0.6607 | **0.5546** | **0.6030** | **0.8464** | **Selected & Registered** |
| **Random Forest** | `n_estimators=100, max_depth=6` | 0.8018 | **0.6821** | 0.4732 | 0.5588 | 0.8440 | Baseline Tree |
| **Gradient Boosting** | `n_estimators=120, lr=0.1, depth=4`| 0.8001 | 0.6667 | 0.4925 | 0.5665 | 0.8382 | Overfits / Lower Recall |

### Model Selection & Registry Promotion Justification
**Logistic Regression** was selected and promoted to the MLflow Model Registry as `TelcoChurnModel`:
- **Highest F1-Score (0.6030)** and highest **Recall (0.5546)**: In telecom churn, false negatives (failing to identify a churning customer) are vastly more expensive than false positives. Logistic Regression captured over 55% of churners compared to only 47% by Random Forest.
- **Highest ROC-AUC (0.8464)**: Demonstrated superior ranking across all decision thresholds.
- **Lifecycle Transition**: The model was registered as `TelcoChurnModel` (Version 1) and programmatically transitioned through `Staging` -> `Production`.

Artifacts logged for every run:
- Serialized Scikit-learn Pipeline (preprocessing + classifier)
- Confusion Matrix Plot (`cm_<model>.png`)
- ROC Curve Plot (`roc_<model>.png`)

---

## 3. Monitoring & Drift Strategy (Evidently AI)

### Reference vs. Current Setup
- **Reference Dataset (70%)**: Historical training-time baseline representing customer demographics and billing patterns.
- **Current Dataset (30%)**: Simulated incoming production data subjected to controlled synthetic drift:
  - **Numerical Drift**: Added random positive offset (+25 to +45) to `MonthlyCharges`.
  - **Categorical Drift**: Skewed `Contract` distribution by oversampling `Month-to-month` contracts to 75%.
  - **Label Drift**: Flipped 8% of ground truth `Churn` labels to test concept drift.

### Evidently Report Findings
The generated report (`reports/evidently_drift_report.html`) captured the drift accurately:
- `MonthlyCharges` flagged for statistically significant drift (Wasserstein distance test, $p < 0.001$).
- `Contract` flagged for categorical distribution shift ($p < 0.001$).
- Target distribution `Churn` drifted by ~8.5%.

### Custom Metrics Logged to MLflow
1. `drift_mean_diff_MonthlyCharges`: **+$34.28** shift in average monthly customer charges.
2. `drift_m2m_churn_shift`: **-0.1056** churn shift specifically within the month-to-month customer cohort.

### Automated Remediation Trigger
If `drift_mean_diff_MonthlyCharges` exceeds \$15.00 or more than 2 key predictive features drift with $p < 0.05$, the system triggers an alert to prevent model degradation and initiates scheduled retraining on newly labeled data.

---

## 4. How to Run (Step-by-Step)

### 1. Environment Setup
```bash
uv sync
```

### 2. Execute Training, MLflow Logging, & Drift Detection
```bash
uv run python train_churn.py
```
This downloads the dataset, runs 3 models, logs metrics and artifacts to MLflow, registers the best model to `Production`, and exports `reports/evidently_drift_report.html`.

### 3. Start Model Serving API
```bash
uv run uvicorn serve_churn:app --port 8000
```

### 4. Test Real-Time Inference
In a separate terminal:
```bash
# Check service health and loaded registry model
curl http://localhost:8000/health

# Send prediction payload
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.50,
    "TotalCharges": 171.00
  }'
```

### 5. Launch MLflow Dashboard
```bash
uv run mlflow ui --port 5000
```
Open `http://localhost:5000` to inspect experiment metrics, side-by-side runs, model artifacts, and registered model versions.
