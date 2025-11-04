import pandas as pd
import joblib
from pathlib import Path

from train import predict_ensemble
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT_DIR = Path(__file__).resolve().parent.parent
FOLDS_DIR = ROOT_DIR / "models" / "final"
DATA_PATH = ROOT_DIR / "data_task1" / "evaluate"

fold_paths = sorted(FOLDS_DIR.glob("xgb_fold*.pkl"))
if not fold_paths:
    raise FileNotFoundError(f"No fold models found in {FOLDS_DIR}")
models = [joblib.load(p) for p in fold_paths]
print(f"Loaded {len(models)} models: {[p.name for p in fold_paths]}")

def load_split(name):
    df = pd.read_csv(DATA_PATH / f"{name}.csv")
    X = df.filter(like="f").values
    y = df["label"].values
    ids = list(zip(df["gene"], df["transcript"], df["position"]))
    return X, y, ids

if __name__ == "__main__":
    X_test, y_test, ids_test = load_split("test")

    y_pred = predict_ensemble(models, X_test)
    auprc = average_precision_score(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_pred)
    print(f"Final Test AUPRC={auprc:.4f}, AUROC={auroc:.4f}")