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
    ids = list(zip(df["transcript"], df["position"]))
    return X, y, ids

def predict_and_save(models, X, ids, out_path):
    """
    models: list of trained models
    X: np.ndarray of features
    ids: list of (transcript_id, position)
    out_path: path to save CSV
    """
    preds = predict_ensemble(models, X)

    # Create DataFrame with transcript_id, position, and score
    df = pd.DataFrame({
        "transcript_id": [tid for tid, pos in ids],
        "transcript_position": [pos for tid, pos in ids],
        "score": preds
    })

    # Sort by transcript_id then position (optional)
    df.sort_values(["transcript_id", "transcript_position"], inplace=True)

    df.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path} ({len(df)} rows)")
    return df

if __name__ == "__main__":
    X_test, y_test, ids_test = load_split("test")

    y_pred = predict_ensemble(models, X_test)
    auprc = average_precision_score(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_pred)
    print(f"Final Test AUPRC={auprc:.4f}, AUROC={auroc:.4f}")
    
    output_path = Path("predictions")  
    out_csv = out_csv = str(output_path / "test_predictions.csv")
    predict_and_save(models, X_test, ids_test, out_csv)
