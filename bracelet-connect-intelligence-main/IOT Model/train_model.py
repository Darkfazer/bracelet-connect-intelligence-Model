"""
Train a simple model to predict SpO2 (%) from Heart Rate (BPM).
Usage:
    python train_model.py --data synthetic_spo2_hr_1000.csv --out model_spo2_rf.joblib
"""
import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib


def make_features(df):
    df = df.copy()
    # ensure numeric
    df['Heart Rate (BPM)'] = pd.to_numeric(df['Heart Rate (BPM)'], errors='coerce')
    df['SpO2 (%)'] = pd.to_numeric(df['SpO2 (%)'], errors='coerce')
    # rolling features on heart rate
    df['hr_roll_mean_5'] = df['Heart Rate (BPM)'].rolling(5, min_periods=1).mean()
    df['hr_roll_std_5'] = df['Heart Rate (BPM)'].rolling(5, min_periods=1).std().fillna(0)
    df['hr_diff'] = df['Heart Rate (BPM)'].diff().fillna(0)
    # timestamp derived
    if 'Timestamp' in df.columns:
        try:
            df['ts'] = pd.to_datetime(df['Timestamp'], errors='coerce')
            df['hour'] = df['ts'].dt.hour.fillna(0)
            df['minute'] = df['ts'].dt.minute.fillna(0)
        except Exception:
            df['hour'] = 0
            df['minute'] = 0
    else:
        df['hour'] = 0
        df['minute'] = 0
    features = ['Heart Rate (BPM)', 'hr_roll_mean_5', 'hr_roll_std_5', 'hr_diff', 'hour', 'minute']
    return df.dropna(subset=['SpO2 (%)']), features


def train(csv_path, out_path, test_size=0.2, random_state=42):
    df = pd.read_csv(csv_path)
    df_clean, features = make_features(df)
    X = df_clean[features].values
    y = df_clean['SpO2 (%)'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print(f"Trained RandomForestRegressor — RMSE: {rmse:.4f}, R2: {r2:.4f}")

    joblib.dump({'model': model, 'features': features}, out_path)
    print(f"Saved model + metadata to {out_path}")
    return model, features


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='synthetic_spo2_hr_1000.csv', help='CSV file input')
    parser.add_argument('--out', default='model_spo2_rf.joblib', help='Output joblib path')
    args = parser.parse_args()
    train(args.data, args.out)
