import json
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import GroupShuffleSplit

from preprocess import parse_json_line, aggregate_9, onehot28, iter_json_lines, build_dataset_from_json_objects

# ---------------- CONFIG ----------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data_task1" / "dataset0.json"
LABEL_PATH = ROOT_DIR / "data_task1" / "data.info.labelled.csv"
MODEL_DIR = ROOT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def make_data_splits(X, y, ids, test_size=0.3, val_size=0.5, random_state=4262):
    mask = ~pd.isna(y)
    X_tr = X[mask]
    y_tr = y[mask].astype(int)

    groups = np.array([
        str(gene) if gene is not None else tid
        for gene, tid, pos in ids
    ])[mask]

    gss = GroupShuffleSplit(test_size=test_size, random_state=random_state)
    train_idx, temp_idx = next(gss.split(X_tr, y_tr, groups=groups))

    gss2 = GroupShuffleSplit(test_size=val_size, random_state=random_state)
    val_idx, test_idx = next(gss2.split(X_tr[temp_idx], y_tr[temp_idx], groups=groups[temp_idx]))

    train_genes = set(groups[train_idx])
    val_genes   = set(groups[temp_idx][val_idx])
    test_genes  = set(groups[temp_idx][test_idx])
    assert len(train_genes & val_genes) == 0, "Gene overlap between train and val!"
    assert len(train_genes & test_genes) == 0, "Gene overlap between train and test!"
    assert len(val_genes & test_genes) == 0, "Gene overlap between val and test!"
    print(f"Split complete: {len(train_genes)} train genes, {len(val_genes)} val genes, {len(test_genes)} test genes")

    return {
        "train": (X[train_idx], y[train_idx], [ids[i] for i in train_idx]),
        "val":   (X[temp_idx][val_idx], y[temp_idx][val_idx], [ids[i] for i in temp_idx[val_idx]]),
        "test":  (X[temp_idx][test_idx], y[temp_idx][test_idx], [ids[i] for i in temp_idx[test_idx]]),
    }
    
    
def train_final_groupkfold_with_oof(X, y, ids, params, n_splits=5):
    groups = np.array([gene if gene is not None else tid for gene, tid, pos in ids])
    gkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=4262)

    models = []
    oof_pred = np.zeros(len(y), dtype=float)
    fold_metrics = []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]

        clf = xgb.XGBClassifier(
            **params,
            tree_method="hist", 
            eval_metric="aucpr", random_state=42 + fold, n_jobs=-1
        )
        clf.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False
        )
        models.append(clf)

        oof_scores = clf.predict_proba(X_va)[:, 1]
        oof_pred[va_idx] = oof_scores

        auprc = average_precision_score(y_va, oof_scores)
        auroc = roc_auc_score(y_va, oof_scores)
        fold_metrics.append((auprc, auroc))
        print(f"Fold {fold}: AUPRC={auprc:.4f}, AUROC={auroc:.4f}")

    oof_auprc = average_precision_score(y, oof_pred)
    oof_auroc = roc_auc_score(y, oof_pred)
    print(f"OOF AUPRC={oof_auprc:.4f}, OOF AUROC={oof_auroc:.4f}")

    return models, oof_pred, fold_metrics

def predict_ensemble(models, X):
    preds = np.zeros(X.shape[0])
    for clf in models:
        preds += clf.predict_proba(X)[:, 1]
    preds /= len(models)
    return preds

if __name__ == "__main__":
    best_params_path = MODEL_DIR / "best_params.json"
    if not best_params_path.exists():
        raise FileNotFoundError(f"Missing best_params.json at {best_params_path}")
    with open(best_params_path, "r") as f:
        best_params = json.load(f)
    print("Loaded best parameters:", best_params)

    df_labels = pd.read_csv(ROOT_DIR / "data_task1" / "data.info.labelled.csv")
    label_dict = {(r.transcript_id, int(r.transcript_position)): int(r.label)
                for r in df_labels.itertuples(index=False)}
    transcript_to_gene = dict(zip(df_labels["transcript_id"], df_labels["gene_id"]))

    print("Parsing JSON...")
    json_iter = iter_json_lines(ROOT_DIR / "data_task1" / "dataset0.json")
    X, y, ids = build_dataset_from_json_objects(json_iter, label_dict, transcript_to_gene)
    print(f"Built dataset: {X.shape}, positives={np.sum(y==1)}, negatives={np.sum(y==0)}")

    splits = make_data_splits(X, y, ids)
    X_train, y_train, ids_train = splits["train"]
    X_val, y_val, ids_val = splits["val"]
    X_test, y_test, ids_test = splits["test"]

    print(f"Train size={len(X_train)}, Val size={len(X_val)}, Test size={len(X_test)}")

    models, oof_pred, fold_metrics = train_final_groupkfold_with_oof(
        X_train, y_train, ids_train, best_params, n_splits=5
    )

    y_pred = predict_ensemble(models, X_val)
    auprc = average_precision_score(y_val, y_pred)
    auroc = roc_auc_score(y_val, y_pred)
    print(f"\nValidation AUPRC={auprc:.4f}, AUROC={auroc:.4f}")

    y_pred_test = predict_ensemble(models, X_test)
    auprc_test = average_precision_score(y_test, y_pred_test)
    auroc_test = roc_auc_score(y_test, y_pred_test)
    print(f"Test AUPRC={auprc_test:.4f}, AUROC={auroc_test:.4f}")

    # Uncomment if needed
    # for i, model in enumerate(models, 1):
    #     path = MODEL_DIR / f"xgb_fold{i}.pkl"
    #     joblib.dump(model, path)
    #     print(f"Saved fold {i} model → {path}")
