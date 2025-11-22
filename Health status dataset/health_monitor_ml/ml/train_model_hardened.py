"""Hardened training: more conservative grid, calibration, and safer saves.

This script performs training with calibration (CalibratedClassifierCV) and stronger regularization.
It uses a smaller grid to keep runtime reasonable.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV


def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    return pd.read_csv(csv_path)


def prepare(df: pd.DataFrame):
    if "body temperature" in df.columns and "body_temperature" not in df.columns:
        df = df.rename(columns={"body temperature": "body_temperature"})
    X = df[["pulse", "body_temperature", "SpO2"]].values
    y = df["Status"].values
    return X, y


def main():
    root = Path(__file__).resolve().parents[2]
    df = load_data(root / 'Health data.csv')
    X, y = prepare(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [5, 10],
        "clf__min_samples_split": [2, 5]
    }

    grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring='f1_macro', n_jobs=-1, refit=True)
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    # calibrate
    calibrated = CalibratedClassifierCV(best.named_steps['clf'], cv=3, method='isotonic')
    # need scaled inputs to fit calibrated classifier
    X_train_scaled = best.named_steps['scaler'].transform(best.named_steps['imputer'].transform(X_train))
    calibrated.fit(X_train_scaled, y_train)

    # For API compatibility, keep scaler + calibrated clf
    clf_path = (Path(__file__).resolve().parent / 'models' / 'health_classifier_hardened.pkl')
    scaler_path = (Path(__file__).resolve().parent / 'models' / 'scaler_hardened.pkl')
    meta_path = (Path(__file__).resolve().parent / 'models' / 'model_metadata_hardened.json')
    joblib.dump(calibrated, clf_path)
    joblib.dump(best.named_steps['scaler'], scaler_path)

    # evaluate
    X_test_scaled = best.named_steps['scaler'].transform(best.named_steps['imputer'].transform(X_test))
    y_pred = calibrated.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    metadata = {
        'model': 'RandomForest_calibrated',
        'test_accuracy': float(acc),
        'test_precision': float(prec),
        'test_recall': float(rec),
        'test_f1': float(f1),
        'best_params': grid.best_params_
    }

    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print('Saved hardened model:', clf_path)
    print('Metrics:', metadata)


if __name__ == '__main__':
    main()
