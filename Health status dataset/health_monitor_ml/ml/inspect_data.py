"""Data inspection helpers: duplicates, leakage checks, and simple plots."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def check_duplicates(df: pd.DataFrame) -> int:
    return int(df.duplicated().sum())


def value_counts_by_status(df: pd.DataFrame):
    return df.groupby('Status').size()


def save_histograms(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    features = [c for c in df.columns if c not in ['Status']]
    for f in features:
        plt.figure()
        sns.histplot(df[f], kde=True)
        plt.title(f)
        plt.tight_layout()
        plt.savefig(out_dir / f"hist_{f}.png")
        plt.close()


def save_pairplot(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.pairplot(df, hue='Status', vars=[c for c in df.columns if c != 'Status'])
    plt.savefig(out_dir / 'pairplot.png')


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]
    csv = root / 'Health data.csv'
    df = load_data(csv)
    print('duplicates:', check_duplicates(df))
    print('counts by status:')
    print(value_counts_by_status(df))
    out = Path(__file__).resolve().parent / 'figures'
    save_histograms(df, out)
    try:
        save_pairplot(df, out)
    except Exception as e:
        print('pairplot failed:', e)
