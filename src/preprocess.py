import json, gzip
import numpy as np
import pandas as pd
from pathlib import Path


def parse_json_line(obj):
    (tid, tdata), = obj.items()
    (pos_key, contexts), = tdata.items()
    pos = int(pos_key) if isinstance(pos_key, str) else pos_key
    (ctx7, reads), = contexts.items()
    arr = np.asarray(reads, dtype=float)
    return tid, pos, ctx7, arr

BASE_IDX = {"A":0, "C":1, "G":2, "T":3}
def onehot28(seq7: str) -> np.ndarray:
    out = np.zeros((7,4), dtype=np.int8)
    s = (seq7 or "").upper()
    for i in range(min(7, len(s))):
        j = BASE_IDX.get(s[i], -1)
        if j >= 0:
            out[i, j] = 1
    return out.ravel()

def aggregate_9(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return np.zeros(45, dtype=np.float32)
    mean = arr.mean(axis=0)
    std  = arr.std(axis=0, ddof=0)
    mn   = arr.min(axis=0)
    mx   = arr.max(axis=0)
    med  = np.median(arr, axis=0)
    return np.concatenate([mean, std, mn, mx, med]).astype(np.float32, copy=False)

NUM_COLS = [
    "dwell_m1","sd_m1","mean_m1",
    "dwell_0","sd_0","mean_0",
    "dwell_p1","sd_p1","mean_p1",
]
FEATURE_NAMES = (
    [f"mean_{c}"   for c in NUM_COLS] +
    [f"std_{c}"    for c in NUM_COLS] +
    [f"min_{c}"    for c in NUM_COLS] +
    [f"max_{c}"    for c in NUM_COLS] +
    [f"median_{c}" for c in NUM_COLS] +
    [f"ctx_{i}"    for i in range(28)]
)


def build_dataset_from_json_objects(json_objects, label_dict, transcript_to_gene):
    """
    json_objects: iterable of parsed per-line dicts
    label_dict: {(transcript_id, position): 0/1}
    Returns X (N,73), y (N,), plus ids for later mapping.
    """
    X_rows, y_rows, ids = [], [], []
    for obj in json_objects:
        tid, pos, ctx7, arr = parse_json_line(obj)
        feats45 = aggregate_9(arr)
        ctx28   = onehot28(ctx7)
        X_rows.append(np.concatenate([feats45, ctx28]))
        y_rows.append(label_dict.get((tid, int(pos)), None))
        gene = transcript_to_gene.get((tid, None))
        ids.append((gene, tid, int(pos)))
    X = np.asarray(X_rows, dtype=np.float32)
    y = np.asarray(y_rows)
    return X, y, ids


def iter_json_lines(path: str):
    """Stream NDJSON from .json or .json.gz"""
    p = Path(path)
    if p.suffix == ".gz":
        with gzip.open(p, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                s = line.strip()
                if s:
                    yield json.loads(s)
    else:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    yield json.loads(s)
                    