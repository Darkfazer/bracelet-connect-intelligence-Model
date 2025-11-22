"""Train models for Health Monitor dataset.

Expect a CSV file named `Health data.csv` in the repository root with columns:
- `pulse`, `body_temperature`, `SpO2`, `Status` (0/1/2)

Produces:
- `ml/models/health_classifier.pkl`
- `ml/models/scaler.pkl`
- `ml/models/model_metadata.json`
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
try:
    from xgboost import XGBClassifier
    _HAS_XGBOOST = True
except Exception:
    XGBClassifier = None
    _HAS_XGBOOST = False
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    df = pd.read_csv(csv_path)
    return df


def prepare_features(df: pd.DataFrame):
    # Accept either 'body_temperature' or 'body temperature'
    cols = list(df.columns)
    # Normalize column names with underscores
    if "body_temperature" not in cols and "body temperature" in cols:
        df = df.rename(columns={"body temperature": "body_temperature"})

    expected = ["pulse", "body_temperature", "SpO2", "Status"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    X = df[["pulse", "body_temperature", "SpO2"]].values
    y = df["Status"].values
    return X, y


def main():
    root = Path(__file__).resolve().parents[2]
    csv_path = root / "Health data.csv"
    print(f"Loading dataset from {csv_path}")
    df = load_data(csv_path)

    print("Basic dataset info:")
    print(df.head())
    print(df.describe())

    # Check missing values
    print("Missing values per column:")
    print(df.isna().sum())

    X, y = prepare_features(df)

    # Simple imputation then scaling
    imputer = SimpleImputer(strategy="mean")
    X_imp = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_imp, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "RandomForest": RandomForestClassifier(random_state=42, class_weight=None),
        "SVM": SVC(probability=True, random_state=42),
        "MLP": MLPClassifier(max_iter=500, random_state=42, early_stopping=True),
    }

    # include XGBoost only when available
    if _HAS_XGBOOST and XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)

    # Use pipeline for imputation/scaling to avoid leakage inside CV
    param_grids = {
        "RandomForest": {"clf__n_estimators": [100, 200], "clf__max_depth": [5, 10, None]},
        "SVM": {"clf__C": [0.1, 1, 10], "clf__kernel": ["rbf"]},
        "MLP": {"clf__hidden_layer_sizes": [(50,), (100,), (50, 25)], "clf__alpha": [0.0001, 0.001]},
    }

    if _HAS_XGBOOST and XGBClassifier is not None:
        param_grids["XGBoost"] = {"clf__n_estimators": [100, 200], "clf__max_depth": [3, 6]}

    best_model = None
    best_name = None
    best_score = -1
    all_results = {}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, estimator in models.items():
        print(f"\nTraining {name}...")
        # build pipeline
        pipeline = Pipeline([("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler()), ("clf", estimator)])
        grid = GridSearchCV(pipeline, param_grids[name], cv=cv, scoring="f1_macro", n_jobs=-1, refit=True)
        grid.fit(X_train, y_train)
        print(f"Best params for {name}: {grid.best_params_}")
        print(f"CV best score: {grid.best_score_}")

        # Evaluate on test set (apply pipeline's transform+predict)
        y_pred = grid.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        print("Test set metrics:")
        print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
        print(classification_report(y_test, y_pred, zero_division=0))
        print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

        all_results[name] = {
            "best_params": grid.best_params_,
            "cv_best_score": float(grid.best_score_),
            "test_accuracy": float(acc),
            "test_precision": float(prec),
            "test_recall": float(rec),
            "test_f1": float(f1),
        }

        if f1 > best_score:
            best_score = f1
            # save best estimator pipeline; we'll extract scaler and clf when persisting
            best_model = grid.best_estimator_
            best_name = name

    print(f"\nBest model: {best_name} with F1={best_score:.4f}")

    # Save artifacts
    models_dir = Path(__file__).resolve().parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "health_classifier.pkl"
    scaler_path = models_dir / "scaler.pkl"
    metadata_path = models_dir / "model_metadata.json"

    # If best_model is a pipeline, separate scaler and classifier for compatibility with API
    if isinstance(best_model, Pipeline):
        clf = best_model.named_steps.get("clf")
        skl_scaler = best_model.named_steps.get("scaler")
    else:
        clf = best_model
        skl_scaler = scaler

    joblib.dump(clf, model_path)
    joblib.dump(skl_scaler, scaler_path)

    metadata = {
        "best_model": best_name,
        "chosen_metric": "f1_macro",
        "best_test_f1": best_score,
        "all_results": all_results,
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved model to {model_path}")
    print(f"Saved scaler to {scaler_path}")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
