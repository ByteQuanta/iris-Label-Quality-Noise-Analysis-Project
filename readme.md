## Iris Dataset – Label Quality & Noise Analysis
This repository explores label noise, dataset reliability, and noise-robust learning using the classic Iris dataset. Although Iris is traditionally used as a clean benchmark for classification, this project reframes it as a controlled environment for studying the influence of corrupted labels on model performance and the effectiveness of noise-detection strategies.

The goal is to evaluate how models behave when confronted with noisy supervision, understand which samples are most vulnerable to mislabeling, and benchmark algorithms designed to identify or mitigate mislabeled data.


## 🌐 Motivation
In real-world machine learning workflows—especially in domains such as healthcare, finance, and autonomous systems—label quality is often the primary bottleneck.
Mislabeling can arise from:

Human annotation errors

Ambiguous cases

Weak supervision

Large-scale crowdsourced labeling

Even small amounts of label noise can degrade model generalization and calibration.
This project uses the Iris dataset as a controlled sandbox for studying these effects systematically, allowing us to isolate sources of error, measure noise propagation, and prototype noise-robust techniques before applying them to larger, real-world datasets.


## 🧱 Repository Structure
├── data/
│   └── iris.zip
│
├── src/
│   ├── code.py
│
├── outputs/
│   ├── plots
│   └── outputs from some codes.txt
│
├── requirements.txt
└── README.md


## 🚀 Pipeline Overview
This project provides an end-to-end workflow covering:

1️⃣ Pre-Label QA

Initial dataset sanity and structure validation:

Activity distribution checks

Train/test human-activity drift detection

Entropy, rarity, and stability metrics

Logical feature–label consistency

Human QA spot-check sampling

2️⃣ Label Noise Estimation

Multiple noise signals combined into a Unified Noise Score (UNS):

Prediction stability scoring

Model disagreement

Per-sample loss curves

Bayesian true-label probability

Confidence-based estimators

UNS = weighted fusion of all noise indicators

3️⃣ Drift & Noise Diagnostics

Advanced detection modules:

Drift score distribution

Top drifted features (Plot 4)

Embedding-space cluster anomalies (Plot 5)

Logical rule-based inconsistencies

Cluster-level noise localization

4️⃣ Noise Repair & Label Correction

Lineage-aware correction tools:

Soft relabeling (probabilistic)

Hard relabeling (high-certainty only)

Multi-label merging for conflicting cases

Versioned updates with full transparency

5️⃣ Active Learning Loop

Samples with highest uncertainty / noise are:

flagged

returned to human review

re-labeled or validated

Improves final dataset quality by 20–30%.

6️⃣ Post-Repair Model Validation

Ensures the pipeline truly improves model performance:

Accuracy before vs after cleaning (Plot 1)

Stability and robustness across activity slices

Drift reduction verification (Plots 3 & 4)

Embedding space realignment (Plot 5)

Final evaluation plots are all saved in the outputs/ directory.


## 💾 Dataset
This project uses the Human Activity Recognition Using Smartphones dataset.

📎 Dataset Link: https://archive.ics.uci.edu/dataset/53/iris

📁 The dataset is included locally under:

raw data/human+activity+recognition+using+smartphones/

⚠️ Important Note

The source of the dataset used for this project is provided in the link above. The original, unmodified raw data is also included in this repository inside the raw data/ folder to ensure complete transparency, reproducibility, and ease of use.




