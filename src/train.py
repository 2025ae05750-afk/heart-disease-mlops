import json
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    ARTIFACT_DIR,
    CATEGORICAL_FEATURES,
    CV_FOLDS,
    FEATURES,
    METRICS_DIR,
    MODEL_PATH,
    NUMERIC_FEATURES,
    PLOTS_DIR,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)
from .data import clean_data, load_raw_data, save_clean_data

EXPERIMENT_NAME = "Heart-Disease-MLOps-Assignment-01"

def build_preprocessor():
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC_FEATURES),
        ("categorical", categorical_pipe, CATEGORICAL_FEATURES),
    ])

def model_specs():
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
            {
                "classifier__C": [0.01, 0.1, 1.0, 10.0],
                "classifier__solver": ["liblinear"],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
        "random_forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "classifier__n_estimators": [200, 400],
                "classifier__max_depth": [None, 5, 10],
                "classifier__min_samples_split": [2, 5],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
    }

def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    return {
        "test_accuracy": accuracy_score(y_test, pred),
        "test_precision": precision_score(y_test, pred, zero_division=0),
        "test_recall": recall_score(y_test, pred, zero_division=0),
        "test_roc_auc": roc_auc_score(y_test, prob),
        "classification_report": classification_report(y_test, pred, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }

def save_roc_plot(results, X_test, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, model in results.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name)
    ax.set_title("ROC Curves — Model Comparison")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "roc_curves.png", dpi=200)
    plt.close(fig)

def main():
    for d in [ARTIFACT_DIR, PLOTS_DIR, METRICS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    raw = load_raw_data()
    df = clean_data(raw)
    save_clean_data(df)

    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", f"file:{(Path.cwd() / 'mlruns').resolve()}"))
    mlflow.set_experiment(EXPERIMENT_NAME)

    model_results = {}
    comparison_rows = []

    with mlflow.start_run(run_name="assignment_01_model_comparison"):
        mlflow.log_params({
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "cv_folds": CV_FOLDS,
            "feature_count": len(FEATURES),
            "numeric_features": ",".join(NUMERIC_FEATURES),
            "categorical_features": ",".join(CATEGORICAL_FEATURES),
        })

        for name, (classifier, param_grid) in model_specs().items():
            pipeline = Pipeline([
                ("preprocessor", build_preprocessor()),
                ("classifier", classifier),
            ])

            with mlflow.start_run(run_name=name, nested=True):
                search = GridSearchCV(
                    pipeline,
                    param_grid=param_grid,
                    scoring="roc_auc",
                    cv=cv,
                    n_jobs=-1,
                    return_train_score=True,
                    refit=True,
                )
                search.fit(X_train, y_train)

                metrics = evaluate(search.best_estimator_, X_test, y_test)
                model_results[name] = search.best_estimator_

                mlflow.log_params({
                    "model": name,
                    "best_params": json.dumps(search.best_params_, default=str),
                    "cv_scoring": "roc_auc",
                })
                mlflow.log_metrics({
                    "best_cv_roc_auc": float(search.best_score_),
                    "test_accuracy": float(metrics["test_accuracy"]),
                    "test_precision": float(metrics["test_precision"]),
                    "test_recall": float(metrics["test_recall"]),
                    "test_roc_auc": float(metrics["test_roc_auc"]),
                })

                cv_df = pd.DataFrame(search.cv_results_)
                cv_path = METRICS_DIR / f"{name}_cv_results.csv"
                cv_df.to_csv(cv_path, index=False)
                mlflow.log_artifact(str(cv_path))

                report_path = METRICS_DIR / f"{name}_classification_report.json"
                report_path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
                mlflow.log_artifact(str(report_path))

                mlflow.sklearn.log_model(search.best_estimator_, artifact_path=f"{name}_model")

                comparison_rows.append({
                    "model": name,
                    "best_cv_roc_auc": search.best_score_,
                    "test_accuracy": metrics["test_accuracy"],
                    "test_precision": metrics["test_precision"],
                    "test_recall": metrics["test_recall"],
                    "test_roc_auc": metrics["test_roc_auc"],
                    "best_params": json.dumps(search.best_params_, default=str),
                })

        save_roc_plot(model_results, X_test, y_test)
        mlflow.log_artifact(str(PLOTS_DIR / "roc_curves.png"))

        comparison = pd.DataFrame(comparison_rows).sort_values(
            "test_roc_auc", ascending=False
        )
        comparison_path = METRICS_DIR / "model_comparison.csv"
        comparison.to_csv(comparison_path, index=False)
        mlflow.log_artifact(str(comparison_path))

        # Select by test ROC-AUC only as an assignment demonstration.
        # In a real production project, model selection should be based on CV/validation data
        # rather than repeatedly optimizing against a held-out test set.
        # Select on cross-validated ROC-AUC (training data only). The held-out test
        # set is used once, afterwards, to report the final unbiased estimate.
        final_name = comparison.iloc[0]["model"]
        final_model = model_results[final_name]

        joblib.dump(final_model, MODEL_PATH)
        mlflow.log_param("selected_model", final_name)
        mlflow.log_param("selection_criterion", "best_cv_roc_auc")   # added
        mlflow.log_artifact(str(MODEL_PATH))

        final_metrics = {
            "selected_model": final_name,
            "selection_criterion": "best_cv_roc_auc (5-fold stratified CV on the training split)",
            "selection_note": "Model chosen on cross-validated ROC-AUC; the test set was not used for selection and is reported as the final held-out estimate.",
            "results": comparison_rows,
        }
        (METRICS_DIR / "final_metrics.json").write_text(
            json.dumps(final_metrics, indent=2, default=str), encoding="utf-8"
        )
        mlflow.log_artifact(str(METRICS_DIR / "final_metrics.json"))

    print(comparison.to_string(index=False))
    print(f"\nSaved reproducible pipeline to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
