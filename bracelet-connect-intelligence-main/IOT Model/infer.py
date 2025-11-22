from pathlib import Path
import pandas as pd
import numpy as np
import joblib


def make_features(df):
    df = df.copy()
    # ensure numeric
    if 'Heart Rate (BPM)' in df.columns:
        df['Heart Rate (BPM)'] = pd.to_numeric(df['Heart Rate (BPM)'], errors='coerce')
    # extract hour/minute from timestamp if present
    ts_col = None
    for c in ['Timestamp', 'timestamp', 'time', 'Time']:
        if c in df.columns:
            ts_col = c
            break
    if ts_col is not None:
        try:
            t = pd.to_datetime(df[ts_col], errors='coerce')
            df['hour'] = t.dt.hour.fillna(0).astype(int)
            df['minute'] = t.dt.minute.fillna(0).astype(int)
        except Exception:
            pass
    # rolling features
    df['hr_roll_mean_5'] = df['Heart Rate (BPM)'].rolling(5, min_periods=1).mean()
    df['hr_roll_std_5'] = df['Heart Rate (BPM)'].rolling(5, min_periods=1).std().fillna(0)
    df['hr_diff'] = df['Heart Rate (BPM)'].diff().fillna(0)
    return df


def load_model(model_path: Path):
    model_meta = joblib.load(model_path)
    # model_meta expected to be dict {'model': model, 'features': features}
    if isinstance(model_meta, dict) and 'model' in model_meta:
        return model_meta['model'], model_meta.get('features')
    # fallback: model file may contain the model object directly
    return model_meta, None


def main():
    file_dir = Path(__file__).resolve().parent
    # try dataset in the same folder as this script first, then fall back to project root
    candidates = [file_dir / 'synthetic_spo2_hr_1000.csv', file_dir.parent / 'synthetic_spo2_hr_1000.csv']
    csv_path = next((p for p in candidates if p.exists()), candidates[0])
    model_path = file_dir / 'model_spo2_rf.joblib'

    if not csv_path.exists():
        print(f"ERROR: dataset not found at {csv_path}")
        return
    if not model_path.exists():
        print(f"ERROR: model artifact not found at {model_path}")
        return

    print('Loading dataset...')
    df = pd.read_csv(csv_path)
    df = make_features(df)

    print('Loading model...')
    model, features = load_model(model_path)
    # default feature list if not bundled
    if features is None:
        features = ['Heart Rate (BPM)', 'hr_roll_mean_5', 'hr_roll_std_5', 'hr_diff']

    # keep only features present in data (some saved feature lists include timestamp-derived cols)
    existing_features = [f for f in features if f in df.columns]
    missing = [f for f in features if f not in df.columns]
    if missing:
        print('Warning: some features from the model are missing in the data and will be ignored:', missing)
    features = existing_features
    if not features:
        print('ERROR: no valid features found in the dataframe. Columns available:', df.columns.tolist())
        return

    # Drop rows where features are missing
    df_feat = df.copy()
    df_feat = df_feat.dropna(subset=features)

    take_n = min(10, len(df_feat))
    if take_n == 0:
        print('No valid rows to predict.')
        return

    X = df_feat[features].iloc[:take_n].values
    print(f'Predicting for first {take_n} rows...')
    preds = model.predict(X)

    out = df_feat.iloc[:take_n].loc[:, features].copy()
    out = out.reset_index(drop=True)
    out['pred_SpO2_%'] = np.round(preds, 3)

    out_path = Path(__file__).resolve().parent / 'predictions.csv'
    out.to_csv(out_path, index=False)
    print(f'Saved predictions to {out_path}')
    print(out.to_string(index=False))


if __name__ == '__main__':
    main()
