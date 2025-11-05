# Prediction of m6A RNA Modifications from Direct RNA-Seq Data

## Project Overview
This project is part of **DSA4262 – Data Science in Genomics (AY2025)**.  
Our goal is to develop a machine learning method to **identify m6A RNA modifications** from direct RNA sequencing data and apply the model to SG-NEx cancer cell lines.  

- **Task 1:** Train and evaluate a machine learning classifier using training data (Hct116 cell line).  
- **Task 2:** Apply the best-performing model to five SG-NEx cancer cell lines (A549, Hct116, HepG2, MCF7, K562).  

---
## Environment Installation
`conda create -n dsa4262`

`conda activate dsa4262`

`pip install -r environment.txt`

---
## How to run the code
### Option 1: Run in Python
- **training and evaluation**: `python src/train.py`
- **prediction using sample test data**: `python src/predict.py`
### Option 2: Run via Docker
- **training and evaluation**: 

`docker build -f dsa4262-model_train.Dockerfile -t dsa4262-train .`

`docker run --rm dsa4262-train`
- **prediction using sample test data**: 

`docker build -f dsa4262-model_predict.Dockerfile -t dsa4262-predict .`

`docker run --rm dsa4262-predict`

---
## Task Description
1. Data Description

The raw dataset `dataset0.json` in the `data_task1` folder contains Nanopore Direct RNA-Seq signal features aligned to reference transcript sequences.
Each line corresponds to one transcript position and stores signal-derived features from all sequencing reads covering that position.

The labeled dataset `data.info.labelled.csv` in the same folder provides a 0,1 label for the corresponding site.

**Structure Overview**

Each line is a nested JSON object:
```
{"ENST00000000233":{"244":{"AAGACCA":[[0.00299,2.06,125.0,0.0177,10.4,122.0,0.0093,10.9,84.1],[0.00631,2.53,125.0,0.00844,4.67,126.0,0.0103,6.3,80.9],[0.00465,3.92,109.0,0.0136,12.0,124.0,0.00498,2.13,79.6],[0.00398,2.06,125.0,0.0083,5.01,130.0,0.00498,3.78,80.4],[0.00664,2.92,120.0,0.00266,3.94,129.0,0.013,7.15,82.2],[0.0103,3.83,123.0,0.00598,6.45,126.0,0.0153,1.09,74.8],[0.00398,3.75,126.0,0.00332,4.3,129.0,0.00299,1.93,81.9],[0.00498,3.93,127.0,0.00398,2.51,131.0,0.0111,3.47,79.4] … }
```

The m6a label for the training data is in the following format:

```
gene_id,transcript_id,transcript_position,label
ENSG00000004059,ENST00000000233,244,0
ENSG00000004059,ENST00000000233,261,0
ENSG00000004059,ENST00000000233,316,0
```
2. Data Preprocessing

To preprocess data, we designed a two-stage preprocessing pipeline consisting of 45 aggregated signal-based features, and 28 sequence-context features.

The detailed function can be found in `src/preprocess.py`

3. Model train, evaluate, predict

We implement an ensemble XGBoost model to balance predictive optimization with controlled variance, thereby mitigating the risk of over fitting.

The train and evaluate script can be found in `src/train.py` The evaluation metric would AUPRC and AUROC


To allow simple prediction, we provides a small test data in the folder `data_task1/evaluate`

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
│       ├── EDA.ipynb                     # Exploratory Data Analysis File
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
---
## Acknowledgement
- Data Description: handout_project2_RNAModifications
