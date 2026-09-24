import os
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset, TargetDriftPreset

DATA_DIR = "data"
DATA_PATH = os.path.join(DATA_DIR, "telco_churn.csv")
REPORTS_DIR = "reports"
DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"


def load_or_download_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(DATA_PATH):
        print(f"Downloading Telco Customer Churn dataset from {DATA_URL}...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    return df


def preprocess_data(df):
    df = df.copy()
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # clean TotalCharges whitespace and fill missing
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # target encoding
    df["Churn_binary"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df


def get_feature_columns(df):
    categorical_cols = [
        col for col in df.select_dtypes(include=["object", "string"]).columns
        if col not in ["Churn", "customerID"]
    ]
    numeric_cols = [
        col for col in df.select_dtypes(include=[np.number]).columns
        if col not in ["Churn_binary"]
    ]
    return categorical_cols, numeric_cols


def build_preprocessor(categorical_cols, numeric_cols):
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ]
    )


def train_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor):
    mlflow.set_experiment("telco-customer-churn")

    models = {
        "LogisticRegression": {
            "model": LogisticRegression(C=0.5, max_iter=1000, random_state=42),
            "params": {"C": 0.5, "max_iter": 1000, "solver": "lbfgs"},
        },
        "RandomForest": {
            "model": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
            "params": {"n_estimators": 100, "max_depth": 6, "criterion": "gini"},
        },
        "GradientBoosting": {
            "model": GradientBoostingClassifier(n_estimators=120, learning_rate=0.1, max_depth=4, random_state=42),
            "params": {"n_estimators": 120, "learning_rate": 0.1, "max_depth": 4},
        },
    }

    results = []

    for name, config in models.items():
        print(f"\n--- Training {name} ---")
        with mlflow.start_run(run_name=name) as run:
            clf = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", config["model"])])
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)
            y_proba = clf.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_proba)

            print(f"Metrics - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, ROC-AUC: {auc:.4f}")

            # log parameters and metrics
            mlflow.log_param("model_family", name)
            for k, v in config["params"].items():
                mlflow.log_param(k, v)

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", auc)

            # generate and log confusion matrix artifact
            cm_fig, ax = plt.subplots(figsize=(5, 4))
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap="Blues")
            ax.set_title(f"Confusion Matrix: {name}")
            cm_path = f"reports/cm_{name}.png"
            os.makedirs("reports", exist_ok=True)
            cm_fig.savefig(cm_path, bbox_inches="tight")
            plt.close(cm_fig)
            mlflow.log_artifact(cm_path, artifact_path="plots")

            # generate and log roc curve artifact
            roc_fig, ax = plt.subplots(figsize=(5, 4))
            RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
            ax.set_title(f"ROC Curve: {name} (AUC={auc:.3f})")
            roc_path = f"reports/roc_{name}.png"
            roc_fig.savefig(roc_path, bbox_inches="tight")
            plt.close(roc_fig)
            mlflow.log_artifact(roc_path, artifact_path="plots")

            # log model with sample input
            input_example = X_test.head(2)
            mlflow.sklearn.log_model(
                sk_model=clf,
                name="model",
                input_example=input_example,
                serialization_format="cloudpickle"
            )

            results.append({
                "name": name,
                "run_id": run.info.run_id,
                "f1": f1,
                "auc": auc,
                "accuracy": acc,
                "pipeline": clf
            })

    return results


def register_and_promote_best_model(results):
    # rank by F1 and ROC-AUC
    best = sorted(results, key=lambda x: (x["f1"], x["auc"]), reverse=True)[0]
    print(f"\nBest Model Selected: {best['name']} (F1: {best['f1']:.4f}, AUC: {best['auc']:.4f}, Run ID: {best['run_id']})")

    model_uri = f"runs:/{best['run_id']}/model"
    model_name = "TelcoChurnModel"

    # register model in MLflow Model Registry
    registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)
    version = registered_model.version
    print(f"Registered {model_name} version {version}")

    client = MlflowClient()
    # transition stage: Staging -> Production
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Staging"
    )
    print(f"Transitioned {model_name} v{version} to Staging")

    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production"
    )
    # also set production alias
    client.set_registered_model_alias(name=model_name, alias="production", version=version)
    print(f"Transitioned {model_name} v{version} to Production")

    return best


def run_evidently_drift_monitoring(df, best_run_id):
    print("\n--- Running Evidently AI Drift Analysis ---")
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # 70/30 split into reference and current data
    ref_df, cur_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df["Churn_binary"])
    ref_df = ref_df.copy()
    cur_df = cur_df.copy()

    # inject synthetic drift into current data
    # 1. Numerical drift: increase MonthlyCharges by +35 offset
    cur_df["MonthlyCharges"] = cur_df["MonthlyCharges"] + np.random.uniform(25, 45, size=len(cur_df))

    # 2. Categorical drift: heavily oversample Month-to-month contracts
    month_to_month_mask = cur_df["Contract"] == "Month-to-month"
    cur_df.loc[~month_to_month_mask, "Contract"] = np.random.choice(
        ["Month-to-month", "One year", "Two year"],
        size=(~month_to_month_mask).sum(),
        p=[0.75, 0.15, 0.10],
    )

    # 3. Target label drift: flip 8% of target labels
    flip_indices = cur_df.sample(frac=0.08, random_state=42).index
    cur_df.loc[flip_indices, "Churn_binary"] = 1 - cur_df.loc[flip_indices, "Churn_binary"]
    cur_df["Churn"] = cur_df["Churn_binary"].map({1: "Yes", 0: "No"})

    # define columns for evidently
    cols_to_monitor = [
        "tenure", "MonthlyCharges", "TotalCharges", "Contract",
        "InternetService", "PaymentMethod", "Churn_binary"
    ]
    ev_ref = ref_df[cols_to_monitor].rename(columns={"Churn_binary": "target"})
    ev_cur = cur_df[cols_to_monitor].rename(columns={"Churn_binary": "target"})

    report = Report(metrics=[
        DataDriftPreset(),
        TargetDriftPreset(),
    ])
    report.run(reference_data=ev_ref, current_data=ev_cur)

    report_path = os.path.join(REPORTS_DIR, "evidently_drift_report.html")
    report.save_html(report_path)
    print(f"Evidently Drift Report saved to {report_path}")

    # calculate custom drift metric
    mean_diff_charges = float(cur_df["MonthlyCharges"].mean() - ref_df["MonthlyCharges"].mean())
    ref_m2m_churn = float(ref_df[ref_df["Contract"] == "Month-to-month"]["Churn_binary"].mean())
    cur_m2m_churn = float(cur_df[cur_df["Contract"] == "Month-to-month"]["Churn_binary"].mean())
    churn_rate_shift_m2m = float(cur_m2m_churn - ref_m2m_churn)

    print(f"Custom Metric - MonthlyCharges Mean Difference: ${mean_diff_charges:.2f}")
    print(f"Custom Metric - Month-to-month Churn Shift: {churn_rate_shift_m2m:.4f}")

    # log drift metrics and HTML artifact to MLflow
    with mlflow.start_run(run_id=best_run_id):
        mlflow.log_metric("drift_mean_diff_MonthlyCharges", mean_diff_charges)
        mlflow.log_metric("drift_m2m_churn_shift", churn_rate_shift_m2m)
        mlflow.log_artifact(report_path, artifact_path="evidently_reports")
        print("Logged drift metrics and HTML report artifact to MLflow run.")


def main():
    raw_df = load_or_download_data()
    clean_df = preprocess_data(raw_df)

    cat_cols, num_cols = get_feature_columns(clean_df)
    preprocessor = build_preprocessor(cat_cols, num_cols)

    feature_cols = cat_cols + num_cols
    X = clean_df[feature_cols]
    y = clean_df["Churn_binary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    results = train_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor)
    best = register_and_promote_best_model(results)
    run_evidently_drift_monitoring(clean_df, best["run_id"])
    print("\nTrack A pipeline execution completed successfully!")


if __name__ == "__main__":
    main()
