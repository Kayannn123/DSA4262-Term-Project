# DSA4262-Term-Project
# Prediction of m6A RNA Modifications from Direct RNA-Seq Data

## Project Overview
This project is part of **DSA4262 – Data Science in Genomics (AY2025)**.  
Our goal is to develop a machine learning method to **identify m6A RNA modifications** from direct RNA sequencing data and apply the model to SG-NEx cancer cell lines.  

- **Task 1:** Train and evaluate a machine learning classifier using training data (Hct116 cell line).  
- **Task 2:** Apply the best-performing model to five SG-NEx cancer cell lines (A549, Hct116, HepG2, MCF7, K562).  

---

## Repo Structure
m6A-prediction/
├── README.md
├── environment.yml  (or requirements.txt)
├── Dockerfile
├── train.py
├── predict.py
├── src/
│   ├── model.py
│   ├── preprocess.py
│   ├── utils.py
│   └── ...
├── test_data/
│   ├── data.json
│   ├── data.info
│   └── expected_output.csv
└── docs/
    ├── INSTALL.md
    └── RUN.md

## Installation & Environment Setup
We recommend using **conda** or **virtualenv**.  

```bash
# clone repository
git clone https://github.com/your-team-repo/m6A-modification.git
cd m6A-modification

# create environment
conda create -n m6a python=3.9 -y
conda activate m6a

# install dependencies
pip install -r requirements.txt
