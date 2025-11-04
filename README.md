# Prediction of m6A RNA Modifications from Direct RNA-Seq Data

## Project Overview
This project is part of **DSA4262 – Data Science in Genomics (AY2025)**.  
Our goal is to develop a machine learning method to **identify m6A RNA modifications** from direct RNA sequencing data and apply the model to SG-NEx cancer cell lines.  

- **Task 1:** Train and evaluate a machine learning classifier using training data (Hct116 cell line).  
- **Task 2:** Apply the best-performing model to five SG-NEx cancer cell lines (A549, Hct116, HepG2, MCF7, K562).  

---
## Environment Installation
`conda create -n dsa4262 python=3.10`

`conda activate dsa4262`

`pip install -r environment.txt`

---
## How to run the code
### Option 1: Run in Python
- **training and evaluation**: `python src/train.py`
- **prediction using sample test data**: `python src/predict.py`
### Option 2: Run via Docker
- **training and evaluation**: 

`docker build -f dsa4262-model_train.Dockerfile -t dsa4262-train`

`docker run --rm dsa4262-train`
- **prediction using sample test data**: 

`docker build -f dsa4262-model_predict.Dockerfile -t dsa4262-train`

`docker run --rm dsa4262-predict`

---
## Repo Structure
```
DSA4262-TermProject/
│
├── data_task1/                            # Raw datasets, metadata, and evaluation samples
│   ├── data.info.labelled.csv
│   ├── dataset0.json
│   ├── dataset1.json
│   ├── dataset2.json
│   ├── dataset3.json
│   │
│   ├── evaluate/                          # Test data for evaluation (used in predict.py)
│   │   └── test.csv
│   │
├── eda_feature_dists/                 # Feature distribution plots and statistics
│       ├── c_dt_mean_hist.png
│       ├── c_mean_mean_hist.png
│       ├── c_sd_mean_hist.png
│       ├── m1_dt_mean_hist.png
│       ├── m1_mean_mean_hist.png
│       ├── m1_sd_mean_hist.png
│       ├── p1_dt_mean_hist.png
│       ├── p1_mean_mean_hist.png
│       ├── p1_sd_mean_hist.png
│       └── feature_distribution_summary.csv
│
├── models/                                # Trained models and metadata
│   ├── best_params.json                   # The best parameter combination we have found
│   ├── metadata.json
│   │
│   └── final/                             # Cross-validation fold models
│       ├── xgb_fold1.pkl
│       ├── xgb_fold2.pkl
│       ├── xgb_fold3.pkl
│       ├── xgb_fold4.pkl
│       └── xgb_fold5.pkl
│
├── predictions/                           # Model predictions for test datasets
│   ├── dataset1_predictions.csv
│   ├── dataset2_predictions.csv
│   ├── dataset3_predictions.csv
│   └── pred_dataset0.csv
│
├── results/                               # Evaluation results on SGNex datasets
│   ├── SGNex_A549_directRNA_*.csv
│   ├── SGNex_Hct116_directRNA_*.csv
│   ├── SGNex_HepG2_directRNA_*.csv
│   ├── SGNex_K562_directRNA_*.csv
│   └── SGNex_MCF7_directRNA_*.csv
│
├── src/                                   # Core source code
│   ├── preprocess.py                      # Feature extraction
│   ├── train.py                           # Model training with XGBoost & Group K-Fold CV
│   └── predict.py                         # Ensemble inference and evaluation
│
├── environment.txt                        # Python dependencies
├── .gitignore                             # Git ignore file
│
├── dsa4262-model_train.Dockerfile          # Dockerfile for model training
├── dsa4262-model_predict.Dockerfile        # Dockerfile for model prediction
│
├── README.md                              # Project documentation
│
├── task1_baseline.ipynb                   # Baseline exploration notebook
├── task1.ipynb                            # Task 1 training notebook
└── task2.ipynb                            # Task 2 analysis notebook

```
