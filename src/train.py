"""Train, tune and compare classifiers with MLflow tracking.

Usage:  python -m src.train
"""
import json
import logging
import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import (  # noqa: E402
    GridSearchCV, StratifiedKFold, cross_validate, train_test_split,
)

from src.config import (  # noqa: E402
    CLEAN_FILE, FEATURES, METRICS_FILE, MODEL_DIR, MODEL_FILE, RANDOM_STATE,
    ROOT, TARGET,
)
from src.features import build_pipeline  # noqa: E402

log = logging.getLogger(__name__)

EXPERIMENT = "heart-disease"
SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]

CANDIDATES = {
    "logistic_regression": (
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        {"model__C": [0.01, 0.1, 1.0, 10.0], "model__penalty": ["l2"]},
    ),
    "random_forest": (
        RandomForestClassifier(random_state=RANDOM_STATE),
        {
            "model__n_estimators": [100, 300],
            "model__max_depth": [None, 4, 8],
            "model__min_samples_leaf": [1, 3],
        },
    ),
}


def evaluate(model, X, y) -> dict:
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
    }


def _plots(model, X, y, name, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(model, X, y, ax=ax, cmap="Blues")
    ax.set_title(f"{name} confusion matrix")
    paths.append(out_dir / f"{name}_confusion_matrix.png")
    fig.tight_layout()
    fig.savefig(paths[-1], dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(model, X, y, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title(f"{name} ROC curve")
    paths.append(out_dir / f"{name}_roc_curve.png")
    fig.tight_layout()
    fig.savefig(paths[-1], dpi=110)
    plt.close(fig)
    return paths


def train(data_path=CLEAN_FILE, tracking_uri=None) -> dict:
    tracking_uri = tracking_uri or os.environ.get(
        "MLFLOW_TRACKING_URI", (ROOT / "mlruns").as_uri())
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT)

    df = pd.read_csv(data_path)
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    plot_dir = ROOT / "report" / "figures"

    results = {}
    for name, (estimator, grid) in CANDIDATES.items():
        with mlflow.start_run(run_name=name):
            search = GridSearchCV(build_pipeline(estimator), grid, cv=cv,
                                  scoring="roc_auc", n_jobs=-1)
            search.fit(X_train, y_train)
            best = search.best_estimator_

            cv_scores = cross_validate(best, X_train, y_train, cv=cv,
                                       scoring=SCORING)
            cv_metrics = {f"cv_{m}": cv_scores[f"test_{m}"].mean()
                          for m in SCORING}
            test_metrics = {f"test_{k}": v
                            for k, v in evaluate(best, X_test, y_test).items()}

            mlflow.log_params({"model_type": name, **{
                k.replace("model__", ""): v
                for k, v in search.best_params_.items()}})
            mlflow.log_metrics({**cv_metrics, **test_metrics})
            for p in _plots(best, X_test, y_test, name, plot_dir):
                mlflow.log_artifact(str(p), artifact_path="plots")
            mlflow.sklearn.log_model(best, artifact_path="model",
                                     input_example=X_test.head(3))

            log.info("%s best=%s cv_roc_auc=%.3f test_roc_auc=%.3f", name,
                     search.best_params_, cv_metrics["cv_roc_auc"],
                     test_metrics["test_roc_auc"])
            results[name] = {
                "model": best,
                "params": search.best_params_,
                "run_id": mlflow.active_run().info.run_id,
                **cv_metrics, **test_metrics,
            }

    # Select on cross-validated ROC-AUC (not the small test set)
    best_name = max(results, key=lambda n: results[n]["cv_roc_auc"])
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(results[best_name]["model"], MODEL_FILE)
    summary = {
        "best_model": best_name,
        "models": {n: {k: v for k, v in r.items() if k != "model"}
                   for n, r in results.items()},
    }
    METRICS_FILE.write_text(json.dumps(summary, indent=2, default=str))
    log.info("Selected %s -> %s", best_name, MODEL_FILE)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    train()
