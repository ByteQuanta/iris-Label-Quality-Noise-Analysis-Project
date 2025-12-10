# ============================================================
# A) PRE LABEL QA
# ============================================================

# ================
# 1. Data Import
# ================
import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import zscore
import numpy.linalg as LA

from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

base_path = os.path.expanduser('~') + '/Desktop/datasets/iris'
data_path = os.path.join(base_path, 'iris.data')   # veya 'bezdekIris.data'

print("Data Path:", data_path)

# Iris veri setini oku
assert os.path.exists(data_path), f"Data not found: {data_path}"
df = pd.read_csv(data_path, header=None)
df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]

# ================
# 2. Basic Validation (Pre-QA entry check)
# ================
X = df.drop("species", axis=1)
y = df["species"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- CRITICAL: Keep original indices for lineage; also create safe working copies when needed
# Resetting indices too early can break lineage when removing rows by original index.
X_train = X_train.copy()
X_test  = X_test.copy()
y_train = y_train.copy()
y_test  = y_test.copy()

# Basic sanity checks
assert X_train.shape[1] == X_test.shape[1], "❌ Train & Test feature mismatch!"
assert y_train.shape[0] == X_train.shape[0], "❌ y_train length mismatch!"
assert y_test.shape[0]  == X_test.shape[0],  "❌ y_test length mismatch!"

print("✅ Data successfully loaded and validated.")
print(f"Train Shape: {X_train.shape}")
print(f"Test Shape : {X_test.shape}")
print(f"# Features : {X_train.shape[1]}")

# ============================================================
# 2) EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
print("\n==============================")
print(" EXPLORATORY DATA ANALYSIS")
print("==============================\n")

print("=== Dataset Overview ===")
print("Train shape:", X_train.shape, " | Test shape:", X_test.shape)
print("y_train shape:", y_train.shape, " | y_test shape:", y_test.shape)
print("\nFeature count:", X_train.shape[1])
print("Unique class count (train):", y_train.nunique())
print("Unique class count (test):", y_test.nunique())

print("\n=== Null / Missing / Zero Row Check ===")
print("Train missing values:", X_train.isnull().sum().sum())
print("Test missing values :", X_test.isnull().sum().sum())
print("Train zero-sum-like rows:", (X_train.sum(axis=1) == 0).sum())
print("Test zero-sum-like rows :", (X_test.sum(axis=1) == 0).sum())

print("\n=== Class Distribution ===")
print("Train class counts:\n", y_train.value_counts())
print("\nTest class counts:\n", y_test.value_counts())

valid_classes = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
invalid_train = y_train[~y_train.isin(valid_classes)]
invalid_test  = y_test[~y_test.isin(valid_classes)]
print("\nInvalid labels (train):", len(invalid_train))
print("Invalid labels (test) :", len(invalid_test))

print("\n=== Feature Integrity Checks ===")
print("Negative value count (train):", (X_train < 0).sum().sum())
print("Negative value count (test) :", (X_test < 0).sum().sum())

feature_variances = X_train.var()
low_var_cols = feature_variances[feature_variances < 1e-6]
print("\nLow-variance feature count:", len(low_var_cols))
print("Low-variance features:\n", low_var_cols)

print("\n=== Zero-Norm Row Detection ===")
row_norm_train = np.linalg.norm(X_train, axis=1)
row_norm_test  = np.linalg.norm(X_test, axis=1)
print("Train norm=0 rows:", (row_norm_train == 0).sum())
print("Test norm=0 rows :", (row_norm_test == 0).sum())

print("\n=== Summary Statistics ===")
print(X_train.describe().T)

print("\n=== Train–Test Feature Mean Comparison ===")
train_means = X_train.mean()
test_means  = X_test.mean()
mean_diff = (train_means - test_means).abs().sort_values(ascending=False)
print("Feature mean differences:\n", mean_diff)
print("\nEDA Completed.\n")

# ============================================================
# 3) Missing Value / Type / Inf checks (short)
# ============================================================
print("=== Checking Missing Value ===")
print("X_train missing values:", X_train.isnull().sum().sum())
print("X_test missing values :", X_test.isnull().sum().sum())
print("y_train missing values:", y_train.isnull().sum().sum())
print("y_test missing values :", y_test.isnull().sum().sum())

print("\n=== Format / Type Control ===")
non_numeric_cols_train = X_train.select_dtypes(exclude=[np.number]).columns
non_numeric_cols_test  = X_test.select_dtypes(exclude=[np.number]).columns
print("Train non-numeric columns:", list(non_numeric_cols_train))
print("Test non-numeric columns :", list(non_numeric_cols_test))

print("\n=== Inf / -Inf Control ===")
print("Train inf count:", np.isinf(X_train).sum().sum())
print("Test inf count :", np.isinf(X_test).sum().sum())

# ============================================================
# 4) Outlier Analysis — IRIS VERSION (NO SUBJECT COLUMN)
# ============================================================
numeric_columns = X_train.select_dtypes(include=[np.number]).columns
# Keep copies that preserve original indices for safe lineage
Xtr = X_train.copy()
Xte = X_test.copy()

print("Train shape:", Xtr.shape)
print("Test shape :", Xte.shape)

# Z-score
z_train = np.abs(zscore(Xtr[numeric_columns]))
# handle 1-D returns from zscore
if z_train.ndim == 1:
    z_train = z_train.reshape(-1, len(numeric_columns))

z_test  = np.abs(zscore(Xte[numeric_columns]))
if z_test.ndim == 1:
    z_test = z_test.reshape(-1, len(numeric_columns))

Xtr["zscore_outliers"] = (z_train > 3).sum(axis=1)
Xte["zscore_outliers"] = (z_test  > 3).sum(axis=1)

# IQR
Q1 = Xtr[numeric_columns].quantile(0.25)
Q3 = Xtr[numeric_columns].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
Xtr["iqr_outliers"] = ((Xtr[numeric_columns] < lower) | (Xtr[numeric_columns] > upper)).sum(axis=1)
Xte["iqr_outliers"] = ((Xte[numeric_columns] < lower) | (Xte[numeric_columns] > upper)).sum(axis=1)

# Mahalanobis
cov = np.cov(Xtr[numeric_columns].values, rowvar=False)
cov_inv = LA.pinv(cov)
mean_vec = Xtr[numeric_columns].mean().values
def mahalanobis_distance(x):
    diff = x - mean_vec
    return np.sqrt(diff @ cov_inv @ diff.T)
md_train = Xtr[numeric_columns].apply(lambda row: mahalanobis_distance(row.values), axis=1)
md_test  = Xte[numeric_columns].apply(lambda row: mahalanobis_distance(row.values), axis=1)
Xtr["mahalanobis_score"] = md_train
Xte["mahalanobis_score"] = md_test

# Model-based
iso = IsolationForest(n_estimators=300, contamination='auto', random_state=42)
iso.fit(Xtr[numeric_columns])
Xtr["iforest_outliers"] = (iso.predict(Xtr[numeric_columns]) == -1).astype(int)
Xte["iforest_outliers"] = (iso.predict(Xte[numeric_columns]) == -1).astype(int)

oc = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
oc.fit(Xtr[numeric_columns])
Xtr["ocsvm_outliers"] = (oc.predict(Xtr[numeric_columns]) == -1).astype(int)
Xte["ocsvm_outliers"] = (oc.predict(Xte[numeric_columns]) == -1).astype(int)

# Robust scaling
scaler = RobustScaler()
# keep DataFrame indices to preserve alignment with outlier flags
Xtr_scaled = pd.DataFrame(scaler.fit_transform(Xtr[numeric_columns]), index=Xtr.index, columns=numeric_columns)
Xte_scaled = pd.DataFrame(scaler.transform(Xte[numeric_columns]), index=Xte.index, columns=numeric_columns)

# Add outlier cols back using original indices (avoid reset_index here)
add_cols = ["zscore_outliers","iqr_outliers","mahalanobis_score","iforest_outliers","ocsvm_outliers"]
Xtr_clean = pd.concat([Xtr_scaled, Xtr[add_cols]], axis=1)
Xte_clean = pd.concat([Xte_scaled, Xte[add_cols]], axis=1)

print("\nFinal cleaned train shape:", Xtr_clean.shape)
print("Final cleaned test shape :", Xte_clean.shape)


# ============================================================
# 5) Duplicate Analysis — IRIS VERSION (TRAIN ONLY)
# ============================================================
# Work on feature-only copies (keep original indices)
Xtr_clean_no_outliers = Xtr_clean.drop(columns=add_cols)
Xte_clean_no_outliers = Xte_clean.drop(columns=add_cols)

print("\nNew Xtr_clean shape:", Xtr_clean_no_outliers.shape)
print("New Xte_clean shape:", Xte_clean_no_outliers.shape)

# Basic duplicate checks
print("\n=== Duplicate Row Detection ===")
print("Train duplicates:", Xtr_clean_no_outliers.duplicated().sum())
print("Test duplicates :", Xte_clean_no_outliers.duplicated().sum())
print("\n=== Feature-only duplicate check ===")
print("Train feature duplicates:", Xtr_clean_no_outliers.duplicated(keep=False).sum())
print("Test feature duplicates :", Xte_clean_no_outliers.duplicated(keep=False).sum())

# Helpers for near-duplicate detection (preserve indices)
def normalize_features(X):
    return StandardScaler().fit_transform(X)

def compressed_representation(X, dim=8):
    dim = min(dim, X.shape[1])
    pca = PCA(n_components=dim)
    return pca.fit_transform(X)

def count_near_duplicates_full(X, threshold=0.9999):
    sim = cosine_similarity(X)
    mask = np.triu(sim > threshold, k=1)
    dup_count = np.sum(mask)
    severity_score = np.sum((sim[mask] - threshold) / (1 - threshold)) if dup_count > 0 else 0.0
    return dup_count, severity_score, sim

def duplicate_groups_full(X, threshold=0.9999, index_values=None):
    # Build similarity matrix and extract connected components (transitive)
    sim = cosine_similarity(X)
    n = sim.shape[0]
    visited = set()
    groups = []
    for i in range(n):
        if i in visited:
            continue
        grp = set([i])
        # BFS to collect transitive neighbors
        queue = [i]
        while queue:
            cur = queue.pop(0)
            for j in range(n):
                if j not in grp and sim[cur, j] > threshold:
                    grp.add(j)
                    queue.append(j)
        if len(grp) > 1:
            visited |= grp
            if index_values is not None:
                groups.append([index_values[g] for g in sorted(list(grp))])
            else:
                groups.append(sorted(list(grp)))
    return groups

# merge_groups and representative selection remain but operate on original indices
def merge_groups(groups):
    merged = []
    used = set()
    for g in groups:
        if any(x in used for x in g):
            continue
        comp = set(g)
        changed = True
        while changed:
            changed = False
            for h in groups:
                if any(x in comp for x in h):
                    before = len(comp)
                    comp |= set(h)
                    if len(comp) != before:
                        changed = True
        used |= comp
        merged.append(sorted(list(comp)))
    return merged

def select_representative_duplicate(X_subset, group_indices):
    df = X_subset.loc[group_indices].copy()
    row_variance = df.var(axis=1)
    rep_idx = row_variance.idxmax()
    return rep_idx

def duplicate_lineage_log(merged_groups, representative_map, severity_score,
                          path_csv="iris_duplicate_lineage.csv",
                          path_json="iris_duplicate_lineage.json"):
    records = []
    ts = datetime.now().isoformat()
    for gid, group in enumerate(merged_groups, start=1):
        rep = representative_map[gid]
        records.append({
            "group_id": gid,
            "timestamp": ts,
            "representative": int(rep),
            "members": list(map(int, group)),
            "group_size": len(group),
            "severity_score": float(severity_score)
        })
    df_lineage = pd.DataFrame(records)
    df_lineage.to_csv(path_csv, index=False)
    with open(path_json, "w") as f:
        json.dump(records, f, indent=4)
    print("\nDuplicate lineage saved:")
    print("CSV :", path_csv)
    print("JSON:", path_json)
    return df_lineage

def detect_near_duplicates_iris(X, threshold=0.9999):
    X_values = X.values
    X_norm = normalize_features(X_values)
    X_pca = compressed_representation(X_norm, dim=8)
    dup_count, severity_score, sim_full = count_near_duplicates_full(X_pca, threshold)
    groups = duplicate_groups_full(X_pca, threshold, index_values=list(X.index))
    merged_groups = merge_groups(groups)
    representative_map = {}
    for gid, group in enumerate(merged_groups, start=1):
        rep = select_representative_duplicate(X, group)
        representative_map[gid] = rep
    print("\n=== Iris Near-Duplicate Analysis ===")
    print("Total duplicate collisions:", dup_count)
    print("Group count:", len(merged_groups))
    print("Severity score:", severity_score)
    return dup_count, severity_score, groups, merged_groups, representative_map

# Run detection on TRAIN features (no outlier columns)
dup_count, severity_score, groups, merged_groups, representative_map = detect_near_duplicates_iris(Xtr_clean_no_outliers)
duplicate_lineage_log(merged_groups, representative_map, severity_score)

# ============================================================
# 11) Clean duplicates from TRAIN dataset (Final, Correct)
# ============================================================
# 1) Final clean feature matrix (no-outlier columns)
X_train_final = Xtr_clean_no_outliers.copy()

# y_train is already available with original indices; DO NOT reset here yet
# We'll drop using original indices and reset after to keep X/y aligned
y_train_final = y_train.copy()

# 2) Determine rows to drop from duplicate groups (groups are in original indices)
rows_to_drop = []
for gid, group in enumerate(merged_groups, start=1):
    representative = representative_map[gid]
    for idx in group:
        if idx != representative:
            rows_to_drop.append(idx)

# dedupe and sort
rows_to_drop = sorted(set(rows_to_drop))

# 3) Drop duplicate rows safely (only on TRAIN) using original indices
# Ensure indices exist in X_train_final before dropping (safety)
rows_to_drop_safe = [r for r in rows_to_drop if r in X_train_final.index]
X_train_final = X_train_final.drop(index=rows_to_drop_safe)
# Align y_train by dropping same original indices
y_train_final = y_train_final.drop(index=rows_to_drop_safe)

# 4) Now reset indices to sequential for downstream ML, but keep mapping if needed
orig_index_mapping = X_train_final.index.to_series().reset_index(drop=True)
X_train_final = X_train_final.reset_index(drop=True)
y_train_final = y_train_final.reset_index(drop=True)

print("\nFinal cleaned X_train shape:", X_train_final.shape)
print("Final cleaned y_train shape:", y_train_final.shape)
print("Removed duplicate row original indices:", rows_to_drop_safe)


# ============================================================
# 6) Correlation Analysis — IRIS VERSION (FINAL AFTER DUPLICATES)
# ============================================================

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ------------------------------------------------------------
# 0) Utility — VIF Calculation (safe)
# ------------------------------------------------------------
def calculate_vif(X):
    vif_df = pd.DataFrame()
    vif_df["feature"] = X.columns
    vif_vals = []
    # handle constant or near-constant columns
    non_const = [c for c in X.columns if X[c].var() > 1e-12]
    for col in X.columns:
        if col not in non_const:
            vif_vals.append(float('inf'))
        else:
            try:
                vif_vals.append(float(variance_inflation_factor(X[non_const].values, non_const.index(col))))
            except Exception:
                vif_vals.append(float('nan'))
    vif_df["VIF"] = vif_vals
    return vif_df.set_index("feature")["VIF"].to_dict()

# ------------------------------------------------------------
# 1) Drift Calculation (Train–Test Distribution Shift)
# ------------------------------------------------------------
def calculate_drift(X_train, X_test):
    drift = {}
    for col in X_train.columns:
        if col in X_test.columns:
            drift[col] = abs(X_train[col].mean() - X_test[col].mean())
        else:
            drift[col] = float('nan')
    return drift

# ------------------------------------------------------------
# 2) Representative Feature Selection
# ------------------------------------------------------------
def select_representative_feature(group, scores):
    df = pd.DataFrame({
        "importance":   {f: scores["importance"].get(f, 0) for f in group},
        "target_corr":  {f: scores["target_corr"].get(f, 0) for f in group},
        "drift":        {f: scores["drift"].get(f, 0) for f in group},
        "vif":          {f: scores["vif"].get(f, 0) for f in group},
        "nan_rate":     {f: scores["nan_rate"].get(f, 0) for f in group},
        "variance":     {f: scores["variance"].get(f, 0) for f in group},
    }).T

    df = df.T  # features as rows

    df = df.sort_values(
        by=[
            "importance",
            "target_corr",
            "drift",
            "vif",
            "nan_rate",
            "variance"
        ],
        ascending=[False, False, True, True, True, False]
    )

    return df.index[0]

# ------------------------------------------------------------
# 3) Main Correlation Analysis
# ------------------------------------------------------------
def correlation_feature_selector(
        X_train_final,
        X_test_clean,
        y_train_final=None,
        importance_scores=None,
        threshold=0.90):

    # ------------------------
    # Default importance scores
    # ------------------------
    if importance_scores is None:
        importance_scores = {col: 0 for col in X_train_final.columns}

    # ------------------------
    # Compute target correlation (NEEDS NUMERIC ENCODING)
    # ------------------------
    from sklearn.preprocessing import LabelEncoder

    if y_train_final is not None:
        # Convert string labels to numeric for correlation
        le = LabelEncoder()
        y_encoded = pd.Series(le.fit_transform(y_train_final), index=y_train_final.index)

        target_corr = X_train_final.apply(lambda c: abs(c.corr(y_encoded)))
        target_corr = target_corr.fillna(0).to_dict()
    else:
        target_corr = {col: 0 for col in X_train_final.columns}


    # ------------------------
    # Additional metrics
    # ------------------------
    drift_scores = calculate_drift(X_train_final, X_test_clean)
    vif_scores = calculate_vif(X_train_final)
    nan_rate = X_train_final.isna().mean().to_dict()
    variance = X_train_final.var().to_dict()

    # ------------------------
    # Correlation groups (connected components - transitive)
    # ------------------------
    corr = X_train_final.corr().abs()

    groups = []
    visited = set()

    cols = corr.columns.tolist()
    # build adjacency
    adj = {c: set() for c in cols}
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            if corr.iloc[i, j] > threshold:
                adj[cols[i]].add(cols[j])
                adj[cols[j]].add(cols[i])

    # BFS for connected components
    for c in cols:
        if c in visited:
            continue
        stack = [c]
        comp = set()
        while stack:
            cur = stack.pop()
            if cur in comp:
                continue
            comp.add(cur)
            visited.add(cur)
            for nb in adj[cur]:
                if nb not in comp:
                    stack.append(nb)
        groups.append(sorted(list(comp)))

    # ------------------------
    # Select best representative feature from each group
    # ------------------------
    keep = []

    for group in groups:
        if len(group) == 1:
            keep.append(group[0])
            continue

        scores = {
            "importance": importance_scores,
            "target_corr": target_corr,
            "drift": drift_scores,
            "vif": vif_scores,
            "nan_rate": nan_rate,
            "variance": variance,
        }

        best = select_representative_feature(group, scores)
        keep.append(best)

    # ------------------------
    # Final reduced dataset
    # ------------------------
    # Align keep with test set columns safely
    common_keep = [c for c in keep if c in X_test_clean.columns]
    missing_in_test = [c for c in keep if c not in X_test_clean.columns]
    if missing_in_test:
        print("Warning: these selected features not present in test and will be dropped:", missing_in_test)

    X_train_reduced = X_train_final[keep].copy()
    X_test_reduced  = X_test_clean[common_keep].copy()

    print("\n=== Correlation Analysis Summary ===")
    print("Original train shape :", X_train_final.shape)
    print("Reduced train shape  :", X_train_reduced.shape)
    print("Original test shape  :", X_test_clean.shape)
    print("Reduced test shape   :", X_test_reduced.shape)
    print("\nSelected features:\n", keep)

    return X_train_reduced, X_test_reduced, groups

# USE
# Ensure test features are aligned to train feature names
common_feats = [c for c in X_train_final.columns if c in Xte_clean_no_outliers.columns]
if not common_feats:
    raise RuntimeError("No common features between train and test after preprocessing!")
Xte_aligned = Xte_clean_no_outliers[common_feats].copy()

X_train_corr, X_test_corr, corr_groups = correlation_feature_selector(
    X_train_final,
    Xte_aligned,     # test clean set aligned
    y_train_final,   # duplicate sonrası target
    importance_scores=None,
    threshold=0.90
)

# Final check
print("\nFINAL: X_train_corr shape:", X_train_corr.shape)
print("FINAL: X_test_corr shape :", X_test_corr.shape)
print("FINAL: y_train_final shape:", y_train_final.shape)

# Ensure alignment
if X_train_corr.shape[0] != y_train_final.shape[0]:
    print("Warning: TRAIN feature rows != target rows — check alignment!")
else:
    print("Index alignment OK for training set.")

# Persist final datasets
X_train_corr.to_csv('X_train_corr.csv', index=False)
X_test_corr.to_csv('X_test_corr.csv', index=False)
y_train_final.to_csv('y_train_final.csv', index=False)

print('\nSaved final reduced datasets to CSV files: X_train_corr.csv, X_test_corr.csv, y_train_final.csv')



# ============================================================
# 7) Class Distribution Analysis (AFTER DUPLICATE CLEANING)
# ============================================================
import matplotlib.pyplot as plt
import seaborn as sns

print("=== Class Distribution Analysis ===\n")

# ------------------------------------------------------------
# 0) Index Alignment (Safety Check)
# ------------------------------------------------------------
# Train target MUST follow the reduced train index
if len(y_train_final) != len(X_train_corr):
    print("[Warning] y_train_final index was not aligned with X_train_corr. Fixing...")
    y_train_final = y_train_final.loc[X_train_corr.index]

# Test target MUST follow reduced test index if shapes differ
if len(y_test) != len(X_test_corr):
    print("[Warning] y_test index was not aligned with X_test_corr. Fixing...")
    y_test = y_test.loc[X_test_corr.index]


# ============================================================
# 1. Basic distribution
# ============================================================
print("Train activity distribution:")
print(y_train_final.value_counts(normalize=True) * 100)

print("\nTest activity distribution:")
print(y_test.value_counts(normalize=True) * 100)

# ============================================================
# 2. Visualization
# ============================================================
train_class_dist = y_train_final.value_counts(normalize=True) * 100
test_class_dist  = y_test.value_counts(normalize=True) * 100

plt.figure(figsize=(12, 6))

# Train distribution barplot
plt.subplot(1, 2, 1)
sns.barplot(x=train_class_dist.index, y=train_class_dist.values)
plt.title("Train Activity Distribution")
plt.xlabel("Activity")
plt.ylabel("Percentage")
plt.xticks(rotation=45)

# Test distribution barplot
plt.subplot(1, 2, 2)
sns.barplot(x=test_class_dist.index, y=test_class_dist.values)
plt.title("Test Activity Distribution")
plt.xlabel("Activity")
plt.ylabel("Percentage")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# ============================================================
# 8) Data Drift & Source Drift Analysis  (Full Feature Scan)
# ============================================================
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import ks_2samp
from statsmodels.stats.multitest import multipletests

# =========================================================================================
# 0) FUNCTION TO COMPUTE VIF (Variance Inflation Factor)
# =========================================================================================
def calculate_vif_full(X_data):
    cols = [c for c in X_data.columns if X_data[c].var() > 0]
    # use fillna(0.0) to ensure numeric matrix
    X_mat = X_data[cols].fillna(0.0).values
    vif_list = []
    for i in range(len(cols)):
        try:
            vif = variance_inflation_factor(X_mat, i)
        except Exception:
            vif = np.nan
        vif_list.append(vif)
    return pd.DataFrame({"feature": cols, "VIF": vif_list}).set_index("feature")

# =========================================================================================
# 1) FUNCTION TO COMPUTE DRIFT BETWEEN TRAIN AND TEST DATA
# =========================================================================================
def calculate_drift(X_train, X_test):
    """
    Calculates the drift between the train and test datasets based on the mean difference.
    Uses intersection of columns to avoid KeyError if test lacks some columns.
    """
    common = [c for c in X_train.columns if c in X_test.columns]
    drift_scores = {col: abs(X_train[col].mean() - X_test[col].mean()) for col in common}
    # for train-only columns make NaN to signal missing in test
    for c in X_train.columns:
        if c not in drift_scores:
            drift_scores[c] = float('nan')
    # debug print (kept as in original)
    print("Drift Scores (Mean Difference):", drift_scores)
    return drift_scores

# =========================================================================================
# 2) FUNCTION FOR COLLECTING ADVANCED DRIFT MEASURES (KS-TEST, PSI, VARIANCE, MEAN)
# =========================================================================================
def calculate_psi(expected, actual, buckets=10, eps=1e-6):
    """PSI using train-based bin edges and smoothing."""
    quantiles = np.linspace(0, 1, buckets + 1)
    try:
        bin_edges = np.unique(np.quantile(expected, quantiles))
        if len(bin_edges) <= 1:
            return 0.0
    except Exception:
        bin_edges = np.linspace(np.min(expected), np.max(expected), buckets + 1)

    exp_counts, _ = np.histogram(expected, bins=bin_edges)
    act_counts, _ = np.histogram(actual, bins=bin_edges)

    exp_perc = np.clip(exp_counts / exp_counts.sum(), eps, None)
    act_perc = np.clip(act_counts / act_counts.sum(), eps, None)

    psi = np.sum((exp_perc - act_perc) * np.log(exp_perc / act_perc))
    return float(psi)

def compute_advanced_drift(X_train, X_test):
    drift_report = []
    for col in X_train.columns:
        train_vals = X_train[col].dropna()
        test_vals  = X_test[col].dropna() if col in X_test.columns else pd.Series(dtype=float)
        if len(train_vals) < 2 or len(test_vals) < 2:
            ks_stat, ks_p = 0.0, 1.0
        else:
            ks_stat, ks_p = ks_2samp(train_vals, test_vals)
        psi_val = calculate_psi(train_vals.values, test_vals.values if len(test_vals)>0 else train_vals.values)
        mean_drift = abs(train_vals.mean() - (test_vals.mean() if len(test_vals)>0 else train_vals.mean()))
        var_drift  = abs(train_vals.var()  - (test_vals.var()  if len(test_vals)>0 else train_vals.var()))
        drift_report.append({
            "feature": col,
            "ks_stat": ks_stat,
            "ks_p": ks_p,
            "psi": psi_val,
            "mean_drift": mean_drift,
            "var_drift": var_drift
        })
    drift_df = pd.DataFrame(drift_report)
    # adjust p-values
    if drift_df["ks_p"].notnull().any():
        _, p_adj, _, _ = multipletests(drift_df["ks_p"].values, method='fdr_bh')
        drift_df["ks_p_adj"] = p_adj
    else:
        drift_df["ks_p_adj"] = np.nan

    # normalize psi/mean/var/ks_stat to 0-1 (safe with epsilon)
    for c in ["psi", "mean_drift", "var_drift", "ks_stat"]:
        drift_df[c + "_norm"] = (drift_df[c] - drift_df[c].min()) / (drift_df[c].max() - drift_df[c].min() + 1e-9)
    
    # create a composite drift score (tunable weights)
    w = {"ks":0.4, "psi":0.3, "mean":0.2, "var":0.1}
    drift_df["composite_drift"] = (
        w["ks"]*drift_df["ks_stat_norm"] +
        w["psi"]*drift_df["psi_norm"] +
        w["mean"]*drift_df["mean_drift_norm"] +
        w["var"]*drift_df["var_drift_norm"]
    )
    return drift_df.sort_values("composite_drift", ascending=False)

# =========================================================================================
# 3) FEATURE SELECTION WITH IMPROVED METRICS (IMPORTANCE, VIF, TARGET CORRELATION, DRIFT)
# =========================================================================================
def select_representative_feature(group, scores):
    # scores is expected to be a dict of dicts: {metric: {feature: value, ...}, ...}
    # Build a DataFrame rows=features, cols=metrics
    metrics = list(scores.keys())
    data = {f: {m: scores[m].get(f, 0) for m in metrics} for f in group}
    df = pd.DataFrame.from_dict(data, orient='index')
    df = df.sort_values(
        by=["importance", "target_corr", "drift", "vif", "nan_rate", "variance"],
        ascending=[False, False, True, True, True, False]
    )
    return df.index[0]

# =========================================================================================
# 4) MAIN FUNCTION TO SELECT CORRELATED FEATURES WITH ADVANCED METRICS
# =========================================================================================
def correlation_feature_selector_with_drift(
    X_train, X_test, y_train, importance_scores=None, target_corr=None, threshold_corr=0.9
):
    if importance_scores is None:
        importance_scores = {col: 0 for col in X_train.columns}
    
    if target_corr is None:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_encoded = le.fit_transform(y_train)
        target_corr = X_train.apply(lambda c: abs(c.corr(pd.Series(y_encoded, index=y_train.index)))).to_dict()

    drift_scores = calculate_drift(X_train, X_test)
    vif_scores = calculate_vif_full(X_train)["VIF"].to_dict()
    nan_rate = {col: X_train[col].isna().mean() for col in X_train.columns}
    variance = {col: X_train[col].var() for col in X_train.columns}

    corr = X_train.corr().abs()

    # Build adjacency for connected components (transitive)
    cols = corr.columns.tolist()
    adj = {c: set() for c in cols}
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            if corr.iloc[i, j] > threshold_corr:
                adj[cols[i]].add(cols[j])
                adj[cols[j]].add(cols[i])

    # BFS/DFS to get connected components
    groups = []
    visited = set()
    for c in cols:
        if c in visited:
            continue
        stack = [c]
        comp = set()
        while stack:
            cur = stack.pop()
            if cur in comp:
                continue
            comp.add(cur)
            visited.add(cur)
            for nb in adj[cur]:
                if nb not in comp:
                    stack.append(nb)
        groups.append(sorted(list(comp)))

    keep = []
    for group in groups:
        if len(group) == 1:
            keep.append(group[0])
            continue
        scores = {
            "importance": {f: importance_scores.get(f, 0) for f in group},
            "target_corr": {f: target_corr.get(f, 0) for f in group},
            "drift": {f: drift_scores.get(f, 0) for f in group},
            "vif": {f: vif_scores.get(f, np.nan) for f in group},
            "nan_rate": {f: nan_rate.get(f, 0) for f in group},
            "variance": {f: variance.get(f, 0) for f in group},
        }
        keep.append(select_representative_feature(group, scores))

    # Align with test columns safely (use intersection)
    common_keep = [c for c in keep if c in X_test.columns]
    missing_in_test = [c for c in keep if c not in X_test.columns]
    if missing_in_test:
        print("Warning: these selected features not present in test and will be dropped:", missing_in_test)

    X_train_selected = X_train[keep].copy()
    X_test_selected  = X_test[common_keep].copy()

    return X_train_selected, X_test_selected, groups

# =======================================================
# APPLY TO FINAL DATASETS
# =======================================================
# Advanced drift metrics
drift_df = compute_advanced_drift(X_train_corr, X_test_corr)

# Feature selection with correlation + drift + VIF
X_train_selected, X_test_selected, correlation_groups = correlation_feature_selector_with_drift(
    X_train_corr, X_test_corr, y_train_final
)

# =======================================================
# Drift inspection
# =======================================================
drift_scores = calculate_drift(X_train_corr, X_test_corr)
sorted_drift = dict(sorted(drift_scores.items(), key=lambda x: x[1], reverse=True))
high_drift_features = {f: s for f, s in drift_scores.items() if (not np.isnan(s)) and s > 1}

print("\nSelected Features after Drift & Correlation Analysis:", X_train_selected.columns.tolist())
print(f"Number of high-drift features (drift > 1): {len(high_drift_features)}")
print("High Drift Features:")
for f, s in high_drift_features.items():
    print(f" - {f}: {s:.4f}")

print("X_train_selected shape:", X_train_selected.shape)
print("X_test_selected shape :", X_test_selected.shape)

# =======================================================
# Drift Analysis – Extended Inspection
# =======================================================

# -------------------------------------------------------
# Safety Check: Ensure both sets have the same columns
# -------------------------------------------------------
common_columns = X_train_corr.columns.intersection(X_test_corr.columns)
if len(common_columns) != X_train_corr.shape[1]:
    missing = set(X_train_corr.columns) - set(common_columns)
    print("\n[Warning] The following train features are missing in test and will be ignored in drift:")
    for m in missing:
        print("  -", m)

X_train_drift = X_train_corr[common_columns]
X_test_drift  = X_test_corr[common_columns]

# 1) Mean drift scores
drift_scores = calculate_drift(X_train_drift, X_test_drift)

print("\n==============================")
print("📊 Mean Drift Score Summary")
print("==============================")
print(f"Total number of features: {len(drift_scores)}")

# 2) Sort drift scores descending
sorted_drift = dict(sorted(drift_scores.items(), key=lambda x: x[1], reverse=True))

# 3) Count high-drift features
high_drift_features = {f: s for f, s in drift_scores.items() if s > 1}
print(f"Number of high-drift features (drift > 1): {len(high_drift_features)}")

print("\nHigh Drift Features (Drift > 1):")
for f, s in high_drift_features.items():
    print(f" - {f}: {s:.4f}")

# 4) Show top N drifted features
TOP_N = 10
print(f"\nTop {TOP_N} Features with Highest Drift:")
for idx, (f, s) in enumerate(list(sorted_drift.items())[:TOP_N]):
    print(f"{idx+1}. {f}: drift={s:.4f}")

# 5) Text-based histogram visualization
print("\n==============================")
print("📉 Drift Distribution (Text Histogram)")
print("==============================")

max_score = max(drift_scores.values()) if len(drift_scores) > 0 else 0

for f, s in list(sorted_drift.items())[:TOP_N]:
    if max_score == 0:
        bar = ""
    else:
        bar = "█" * int((s / max_score) * 40)
    print(f"{f:25} | {bar} ({s:.3f})")

# 6) If advanced drift (PSI, KS) is available
try:
    if 'drift_df' in locals():
        print("\n==============================")
        print("🔎 Integrating Advanced Drift Metrics (PSI, KS-Drift)")
        print("==============================")
        
        merged = drift_df.set_index("feature").copy()
        merged["mean_drift"] = merged.index.map(lambda x: drift_scores.get(x, float('nan')))
        merged = merged.sort_values(by="ks_stat", ascending=False)

        print("\nTop features by KS Drift:")
        print(merged[["ks_stat", "psi", "mean_drift"]].head(10))

except Exception as e:
    print("\nAdvanced drift metrics could not be merged:", e)

# =======================================================
# Aligning X_test with the selected features
# =======================================================
final_features = X_train_selected.columns

# Safety: ensure test contains all selected features
missing_in_test = [f for f in final_features if f not in X_test_corr.columns]

if len(missing_in_test) > 0:
    print("\n[Warning] The following selected features are missing in X_test_corr and will be dropped:")
    for m in missing_in_test:
        print("  -", m)

safe_features = [f for f in final_features if f in X_test_corr.columns]

X_test_selected = X_test_corr[safe_features]

print("X_test_selected shape:", X_test_selected.shape)
print("X_train_selected shape:", X_train_selected.shape)
print("Final Feature List:", list(safe_features))


# =======================================================
# B) Label Quality
# =======================================================
# 1) Evaluating Label Quality
# =======================================================
import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.model_selection import KFold
from collections import Counter

# =======================================================
# Ensure labels are 1D (keep as numpy arrays for indexing but also keep Series when needed)
# =======================================================
y_train_final = np.asarray(y_train_final).ravel()  # Ensure it's 1D
y_test  = np.asarray(y_test).ravel()

# =======================================================
# 1. Entropy Analysis
# =======================================================
def label_entropy(y):
    # Use pandas value_counts (works for numeric or object labels)
    s = pd.Series(y)
    probs = s.value_counts(normalize=True).values
    return float(entropy(probs))

# Entropi hesaplaması yapılabilir
print("Train Label Entropy:", label_entropy(y_train_final))  # Using y_train_final
print("Test  Label Entropy:", label_entropy(y_test))

# =======================================================
# 2. Rare-Class Noise Risk
# =======================================================
rare_threshold = 0.01  # Less than 1% risk
class_freq = pd.Series(y_train_final).value_counts(normalize=True)  # Using y_train_final
rare_classes = class_freq[class_freq < rare_threshold]

print("\nRare classes (high noise risk):")
print(rare_classes)

# =======================================================
# 3. Train vs Test Unique Label Consistency
# =======================================================
train_unique = set(pd.Series(y_train_final).unique())  # Using y_train_final
test_unique  = set(pd.Series(y_test).unique())

print("\nClasses in TRAIN but not TEST:", train_unique - test_unique)
print("Classes in TEST but not TRAIN:", test_unique - train_unique)

# =======================================================
# 4. Label Adjacency Anomaly Detection
# =======================================================
unique_classes = np.unique(y_train_final)  # Using y_train_final
class_positions = {c: i for i, c in enumerate(unique_classes)}

diffs = []
for i in range(1, len(y_train_final)):  # Using y_train_final
    prev = y_train_final[i-1]
    cur  = y_train_final[i]
    # If a label is unseen in mapping, skip to avoid KeyError
    if prev in class_positions and cur in class_positions:
        diffs.append(abs(class_positions[cur] - class_positions[prev]))
if len(diffs) > 0:
    avg_label_jump = np.mean(diffs)
else:
    avg_label_jump = 0.0
print("\nAverage label adjacency jump:", avg_label_jump)

# =======================================================
# 2) Label Noise Estimation (Final Version)
# =======================================================
# 1. Model Train + Probability Predictions
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train_selected, y_train_final)  # Using y_train_final

# train_proba corresponds to clf.classes_ ordering
train_proba = clf.predict_proba(X_train_selected)
test_proba  = clf.predict_proba(X_test_selected)

# =======================================================
# IMPORTANT: map labels → model class indices (reference order)
# =======================================================
ref_classes = list(clf.classes_)
ref_index = {c: i for i, c in enumerate(ref_classes)}
class_to_index = ref_index
# Map training labels to reference indices (safety: handle unseen labels)
y_train_mapped = np.array([class_to_index.get(y, -1) for y in y_train_final])
if np.any(y_train_mapped == -1):
    raise ValueError("Some training labels were not found in classifier classes_ mapping.")

# =======================================================
# 2. Prediction Stability
# =======================================================
kf = KFold(n_splits=5, shuffle=True, random_state=42)
stability_scores = np.zeros(len(y_train_final))  # Using y_train_final

# Stability Calculation: Tracking consistency across folds
for t_idx, v_idx in kf.split(X_train_selected):
    X_tr = X_train_selected.iloc[t_idx]
    X_val = X_train_selected.iloc[v_idx]
    y_tr  = np.asarray(y_train_final)[t_idx]

    clf_cv = RandomForestClassifier(n_estimators=200, random_state=42)
    clf_cv.fit(X_tr, y_tr)

    val_proba = clf_cv.predict_proba(X_val)
    # use max predicted probability per sample as stability proxy
    stability_scores[v_idx] = np.max(val_proba, axis=1)

Prediction_Stability_noise_score = 1 - stability_scores

# =======================================================
# 3. Model Disagreement
# =======================================================
models = [
    RandomForestClassifier(n_estimators=200, random_state=42),
    ExtraTreesClassifier(n_estimators=200, random_state=42),
    GradientBoostingClassifier()
]

proba_list = []
for m in models:
    m.fit(X_train_selected, y_train_final)
    prob = m.predict_proba(X_train_selected)
    # Align this model's class order to reference order (clf.classes_)
    aligned = np.zeros((prob.shape[0], len(ref_classes)))
    # build mapping from this model's classes_ to indices
    try:
        m_classes = list(m.classes_)
    except Exception:
        # Some ensemble wrappers may not have classes_ attribute in edge cases
        m_classes = ref_classes
    for j, cls in enumerate(m_classes):
        if cls in ref_index:
            aligned[:, ref_index[cls]] = prob[:, j]
        else:
            # unseen class in this model (very unlikely), leave zeros
            pass
    proba_list.append(aligned)

# stack: shape (n_models, n_samples, n_classes)
proba_stack = np.stack(proba_list, axis=0)
# disagreement per sample: std over models, average across classes
Model_Disagreement_score = proba_stack.std(axis=0).mean(axis=1)

# =======================================================
# 4. Loss-Based Noise
# =======================================================
# Ensure train_proba columns are in reference order (they are, since from clf)
loss_per_sample = -np.log(np.clip(train_proba[np.arange(len(y_train_final)), y_train_mapped], 1e-12, 1.0))
# normalize to 0-1
Loss_Based_noise_score = (loss_per_sample - loss_per_sample.min()) / (loss_per_sample.max() - loss_per_sample.min() + 1e-12)

# =======================================================
# 5. Bayesian Confidence
# =======================================================
p_true = train_proba[np.arange(len(y_train_final)), y_train_mapped]
bayesian_confidence_noise_score = 1 - p_true

# =======================================================
# 6. Flag + Continuous Score
# =======================================================
confidence_threshold = 0.5
cleanlab_flag = (p_true < confidence_threshold).astype(int)

# === CleanLab continuous noise score ===
try:
    from cleanlab.internal.multiclass_utils import get_normalized_probs
    probs_norm = get_normalized_probs(train_proba)
    cleanlab_continuous = 1 - probs_norm[np.arange(len(y_train_final)), y_train_mapped]
except Exception:
    cleanlab_continuous = cleanlab_flag.astype(float)

# =======================================================
# 7. Final Unified Noise Score
# =======================================================
noise_df = pd.DataFrame({
    "Prediction_Stability_noise_score": Prediction_Stability_noise_score,
    "Model_Disagreement_score": Model_Disagreement_score,
    "Loss_Based_noise_score": Loss_Based_noise_score,
    "bayesian_confidence_noise_score": bayesian_confidence_noise_score,
    "cleanlab_flag": cleanlab_flag,
    "cleanlab_continuous_score": cleanlab_continuous
})

# **Including Continuous CleanLab, average**
noise_df["final_noise_score"] = noise_df[
    [
        "Prediction_Stability_noise_score",
        "Model_Disagreement_score",
        "Loss_Based_noise_score",
        "bayesian_confidence_noise_score",
        "cleanlab_continuous_score"
    ]
].mean(axis=1)

# =======================================================
# 8. FINAL NOISE REPORT
# =======================================================
noise_df_formatted = noise_df.copy()
noise_df_formatted["sample_id"] = noise_df_formatted.index

# Columns order
noise_df_formatted = noise_df_formatted[
    [
        "sample_id",
        "Prediction_Stability_noise_score",
        "Model_Disagreement_score",
        "Loss_Based_noise_score",
        "bayesian_confidence_noise_score",
        "cleanlab_flag",
        "cleanlab_continuous_score",
        "final_noise_score"
    ]
]

# Normalize final score with error handling for zero division
final_noise_min = noise_df_formatted["final_noise_score"].min()
final_noise_max = noise_df_formatted["final_noise_score"].max()

# Check if normalization range is valid
if final_noise_max > final_noise_min:
    noise_df_formatted["final_noise_score"] = (
        (noise_df_formatted["final_noise_score"] - final_noise_min)
        / (final_noise_max - final_noise_min)
    )
else:
    noise_df_formatted["final_noise_score"] = 0.0  # All values are the same

# Most noisy 20 samples
top_noisy_samples = noise_df_formatted.sort_values("final_noise_score", ascending=False).head(20)

# Display final output
print("\nTop noisy samples (top 20):")
print(top_noisy_samples)



# =======================================================
# 3) NOISE DIAGNOSTICS
# =======================================================
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

# Data from previous sections
X = X_train_selected.copy()

# FIX: y_train_final is numpy array → convert to Series before reset_index
y = pd.Series(y_train_final).reset_index(drop=True)

# FIX: noise score alignment
noise_scores = noise_df["final_noise_score"].reset_index(drop=True)

# Safety alignment
assert len(X) == len(y) == len(noise_scores), "Length mismatch in X, y, or noise_scores!"

# =======================================================
# 1. CLASS-LEVEL NOISE CONCENTRATION
# =======================================================
class_noise_df = pd.DataFrame({
    "label": y,
    "noise": noise_scores
})

class_noise_summary = (
    class_noise_df.groupby("label")["noise"]
    .mean()
    .sort_values(ascending=False)
)

print("\n===== CLASS-LEVEL NOISE CONCENTRATION =====")
print(class_noise_summary)


# =======================================================
# 2. FEATURE-SPACE NOISE CLUSTERING (AUTO CLUSTER COUNT)
# =======================================================
optimal_clusters = max(2, min(10, X.shape[0] // 10))

kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=20)
cluster_labels = kmeans.fit_predict(X)

cluster_noise = (
    pd.DataFrame({"cluster": cluster_labels, "noise": noise_scores})
    .groupby("cluster")["noise"]
    .mean()
    .sort_values(ascending=False)
)

print("\n===== CLUSTER-LEVEL NOISE =====")
print(cluster_noise)


# =======================================================
# 3. PCA EMBEDDING + LOF OUTLIER DETECTION
# =======================================================
n_components = max(2, min(10, X.shape[1] - 1))
pca = PCA(n_components=n_components, random_state=42)

X_embed = pca.fit_transform(X)

lof = LocalOutlierFactor(
    n_neighbors=min(20, len(X) - 1),
    contamination=min(0.03, (len(X) - 1) / len(X))
)

lof_scores = -lof.fit_predict(X_embed)
lof_scores = pd.Series(lof_scores)


embedding_noise_df = pd.DataFrame({
    "lof_score": lof_scores,
    "noise_score": noise_scores
})

print("\n===== PCA–EMBEDDING OUTLIER OVERLAP =====")
print(embedding_noise_df.corr())


# =======================================================
# 4. RULE VIOLATION DETECTOR (DECISION TREE)
# =======================================================
rule_model = DecisionTreeClassifier(
    max_depth=6,
    min_samples_leaf=5,
    random_state=42
)

rule_model.fit(X, y)

rule_pred = rule_model.predict(X)
rule_violations = (rule_pred != y).astype(int)

rule_violation_df = pd.DataFrame({
    "label": y,
    "rule_violation": rule_violations,
    "noise_score": noise_scores
})

print("\n===== RULE-BASED LABEL VIOLATIONS =====")
print(rule_violation_df.groupby("label")[["rule_violation", "noise_score"]].mean())


# =======================================================
# 5. Unified Noise Diagnostics Dashboard
# =======================================================
diagnostics_df = pd.DataFrame({
    "true_label": y,
    "cluster": cluster_labels,
    "noise_score": noise_scores,
    "rule_violation": rule_violations,
    "lof_outlier": lof_scores,
})

scaler = MinMaxScaler()

diagnostics_df[["noise_score", "rule_violation", "lof_outlier"]] = (
    scaler.fit_transform(diagnostics_df[["noise_score", "rule_violation", "lof_outlier"]])
)

diagnostics_df["suspicious_score"] = (
    diagnostics_df["noise_score"] * 0.5 +
    diagnostics_df["lof_outlier"] * 0.3 +
    diagnostics_df["rule_violation"] * 0.2
)

print("\n===== TOP 20 SUSPICIOUS SAMPLES (Unified Diagnostics) =====")
top_suspicious_samples = diagnostics_df.sort_values("suspicious_score", ascending=False).head(20)
print(top_suspicious_samples)



# ============================================================
# 4) DATA LINEAGE & SENSOR PROVENANCE (For Iris Dataset)
# ============================================================
import pandas as pd
import numpy as np
import hashlib
from datetime import datetime
import json
from sklearn.preprocessing import LabelEncoder

# ============================================================
# 1) LINEAGE CONFIG (Annotation for Iris Dataset)
# ============================================================

LINEAGE_CONFIG = {
    "annotator_id": "Iris-Original",
    "annotation_tool_version": "v1.0",
    "guideline_version": "Iris-v1.0",
    "revision_count": 0,
    "agreement_score": 1.0,
    "annotation_source": "Original dataset labels",
}

# ============================================================
# 2) DEVICE ID HASHING (Not applicable for Iris dataset)
# ============================================================

def hash_string(x: str):
    return hashlib.sha256(x.encode()).hexdigest()[:16]

device_hash = hash_string("iris_dataset_device")

# ============================================================
# 3) Build Lineage Table for ALL SAMPLES
# ============================================================

def build_lineage_table(X, y, noise_scores=None):
    """
    X  → feature dataframe (train or test)
    y  → labels (np array or pd series)
    noise_scores → Noise score metric (optional)
    """

    # --- FIX: Convert y to pandas Series for safe indexing ---
    if isinstance(y, np.ndarray):
        y = pd.Series(y)
    else:
        y = y.reset_index(drop=True)

    # --- Noise Score Length Safety ---
    if noise_scores is None:
        noise_scores = np.zeros(len(X))
    else:
        noise_scores = pd.Series(noise_scores).reset_index(drop=True)

    # Align lengths cleanly
    assert len(X) == len(y) == len(noise_scores), "X, y, and noise_scores length mismatch!"

    sample_ids = list(X.index)
    lineage_records = []

    # Compute timestamp once, not for every row
    timestamp_now = datetime.now().isoformat()

    # LabelEncoder for numeric label consistency
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    for i, sid in enumerate(sample_ids):
        lineage_records.append({
            "sample_id": int(sid),
            "label": int(y_encoded[i]),

            # LINEAGE FIELDS
            "annotator_id": LINEAGE_CONFIG["annotator_id"],
            "annotation_timestamp": timestamp_now,
            "annotation_tool_version": LINEAGE_CONFIG["annotation_tool_version"],
            "guideline_version": LINEAGE_CONFIG["guideline_version"],
            "revision_count": LINEAGE_CONFIG["revision_count"],
            "agreement_score": LINEAGE_CONFIG["agreement_score"],
            "annotation_source": LINEAGE_CONFIG["annotation_source"],

            # NOISE SCORE
            "noise_score": float(noise_scores[i]),

            # DEVICE HASH
            "device_hash": device_hash
        })

    return pd.DataFrame(lineage_records)

# ============================================================
# 4) CREATE TRAIN & TEST LINEAGE TABLES
# ============================================================

train_lineage = build_lineage_table(
    X_train_selected,
    y_train_final,
    noise_scores=noise_df["cleanlab_continuous_score"]
)

test_lineage = build_lineage_table(
    X_test_selected,
    y_test,
    noise_scores=np.zeros(len(y_test))
)

# ============================================================
# 5) SAVE ARTIFACTS
# ============================================================

train_lineage.to_csv("train_lineage_iris.csv", index=False)
test_lineage.to_csv("test_lineage_iris.csv", index=False)

with open("sensor_metadata_iris.json", "w") as f:
    json.dump({"sensor_metadata": "Not applicable for Iris dataset"}, f, indent=4)

print("=== LINEAGE + SENSOR PROVENANCE TABLES SAVED ===")
print("train_lineage_iris.csv")
print("test_lineage_iris.csv")
print("sensor_metadata_iris.json")



# ============================================================
# E) GUIDELINE DRIFT ANALYSIS (For Iris Dataset)
# ============================================================

import numpy as np
import pandas as pd

# ============================================================
# 0) Add a timestamp to the train_lineage table if it doesn't exist
# ============================================================
if "timestamp" not in train_lineage.columns:
    train_lineage["timestamp"] = pd.date_range(
        start="2020-01-01",
        periods=len(train_lineage),
        freq="H"
    )


# ============================================================
# 1) Guideline Version Drift
# ============================================================
print("\n==============================")
print("📘 Guideline Version Distribution")
print("==============================")
print(train_lineage["guideline_version"].value_counts())

guideline_drift = (
    train_lineage.groupby("guideline_version")["noise_score"]
    .mean()
    .sort_values(ascending=False)
)

print("\nAverage Noise Score per Guideline Version:")
print(guideline_drift)


# ============================================================
# 2) Annotator Drift
# ============================================================
print("\n==============================")
print("🧑‍🏫 Annotator Drift Summary")
print("==============================")

annotator_drift = (
    train_lineage.groupby("annotator_id")["noise_score"]
    .mean()
    .sort_values(ascending=False)
)

print("Average noise per annotator:")
print(annotator_drift)


# ============================================================
# 3) Time-based Drift
# ============================================================
print("\n==============================")
print("⏳ Time Drift (Chronological)")
print("==============================")

train_lineage["ts_numeric"] = train_lineage["timestamp"].astype("int64")

# Rolling window drift
train_lineage["rolling_noise"] = (
    train_lineage["noise_score"]
    .rolling(window=50, min_periods=5)
    .mean()
)

print("Time-based drift computed (rolling mean).")


# ============================================================
# 4) Device-level Drift
# ============================================================
print("\n==============================")
print("🔧 Device Drift Summary")
print("==============================")

device_drift = (
    train_lineage.groupby("device_hash")["noise_score"]
    .mean()
    .sort_values(ascending=False)
)

print(device_drift)


# ============================================================
# 5) Sensor Noise vs Label Noise Correlation
# ============================================================

# If there is no sensor column, let's add it automatically.
if "sensor_noise_std" not in train_lineage.columns:
    train_lineage["sensor_noise_std"] = 0.0

print("\n==============================")
print("📡 Sensor Noise ↔ Label Noise Correlation")
print("==============================")

corr_val = np.corrcoef(
    train_lineage["sensor_noise_std"],
    train_lineage["noise_score"]
)[0, 1]

print("Correlation:", corr_val)


# ============================================================
# 6) Multidimensional Drift Table
# ============================================================
print("\n==============================")
print("📊 Drift Pivot Table (Guideline × Annotator)")
print("==============================")

pivot_table = train_lineage.pivot_table(
    index="guideline_version",
    columns="annotator_id",
    values="noise_score",
    aggfunc="mean"
)

print(pivot_table)


# ============================================================
# 7) Final drift object
# ============================================================
print("\n==============================")
print("📦 Final Drift Components Assembled")
print("==============================")

drift_components = {
    "guideline_drift": guideline_drift,
    "annotator_drift": annotator_drift,
    "device_drift": device_drift,
    "time_drift": train_lineage["rolling_noise"],
    "sensor_vs_noise_corr": corr_val,
    "pivot_guideline_annotator": pivot_table,
    "lineage_table": train_lineage
}

print("✓ All drift components assembled successfully.")



# =========================== 
# F) ACTIVE LEARNING LOOP
# ===========================
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

# ---------------------------------------
# 0) Ensure prerequisites & choose model
# ---------------------------------------
# models list must exist (from earlier sections)
if "models" not in globals():
    raise RuntimeError("The list of models could not be found. Please load the models.")

# Use first model (RandomForest expected) as active-learning model
model = models[0]

# Ensure lineage table reference: prefer lineage_train if present else train_lineage
if "lineage_train" in globals():
    lineage_df = train_lineage
elif "train_lineage" in globals():
    lineage_df = train_lineage
else:
    # Create a minimal lineage_df if none exists (fallback)
    lineage_df = pd.DataFrame({"sample_id": np.arange(len(y_train_final))})
    lineage_df["annotator_id"] = "annotator_1"
    lineage_df["revision_id"] = 0
    lineage_df["annotation_timestamp"] = pd.Timestamp.now()

# ---------------------------------------
# Prepare y_train_final as a pandas Series for safe in-place updates
# ---------------------------------------
if isinstance(y_train_final, np.ndarray):
    y_series = pd.Series(y_train_final).reset_index(drop=True)
else:
    y_series = pd.Series(y_train_final).reset_index(drop=True)

# Ensure X_train_selected is a DataFrame and indices align
X_sel = X_train_selected.copy().reset_index(drop=True)

# ---------------------------------------
# Fit model if not already fitted
# ---------------------------------------
if not hasattr(model, "classes_"):
    print("Model daha önce eğitilmemiş — şimdi eğitiliyor...")
    model.fit(X_sel, y_series.values)
else:
    print("Eğitilmiş model bulundu — yeniden eğitim yapılmadı.")

# ---------------------------------------
# 1) UNCERTAINTY SCORES
# ---------------------------------------
probs = model.predict_proba(X_sel)
# least confidence
least_conf = 1 - probs.max(axis=1)
# entropy
entropy_vals = -(probs * np.log(probs + 1e-12)).sum(axis=1)

# losses per-sample (safe: clip probs and map labels)
# map training labels to model.classes_ indices for log_loss call
# we'll compute single-sample log_loss using sklearn (labels argument ensures class order)
losses = np.zeros(len(y_series))
for i in range(len(y_series)):
    try:
        losses[i] = log_loss([y_series.iloc[i]], [probs[i]], labels=list(model.classes_))
    except Exception:
        # fallback to high loss for safety
        losses[i] = np.max([0.0, np.nan_to_num(losses[i])])

# ---------------------------------------
# 2) ANNOTATOR DISAGREEMENT
# ---------------------------------------
# If lineage has multiple annotator votes per sample, compute disagreement;
# otherwise (single annotator) set disagreement to 0.
if "annotator_id" in lineage_df.columns:
    # if multiple rows per sample_id exist, compute disagreement per sample_id
    if lineage_df["sample_id"].duplicated().any():
        # compute for each sample: number of unique labels assigned / num annotators
        # requires 'true_label' or 'annotation_label' column; if not present, fall back to counts
        if "true_label" in lineage_df.columns:
            ann_group = lineage_df.groupby("sample_id")["true_label"].nunique()
            ann_total = lineage_df.groupby("sample_id")["annotator_id"].nunique()
            annotator_disagreement_full = (ann_group / ann_total).reindex(range(len(y_series))).fillna(0.0)
            # disagreement measure: 1 - consensus
            annotator_disagreement = 1 - annotator_disagreement_full.values
        else:
            # fallback: use fraction of distinct annotators (no label info)
            annotator_counts = lineage_df.groupby("sample_id")["annotator_id"].nunique().reindex(range(len(y_series))).fillna(1)
            annotator_disagreement = 1 - (annotator_counts / annotator_counts.max()).values
    else:
        # single annotator per sample → no disagreement
        annotator_disagreement = np.zeros(len(y_series))
else:
    annotator_disagreement = np.zeros(len(y_series))

# ---------------------------------------
# 3) CLEANLAB SCORE (safe read)
# ---------------------------------------
if "noise_df" in globals() and "cleanlab_continuous" in noise_df.columns:
    cleanlab_score = pd.Series(noise_df["cleanlab_continuous"]).reset_index(drop=True).values
else:
    cleanlab_score = np.zeros(len(y_series))

# ---------------------------------------
# 4) UNIFIED ACTIVE SCORE (safe normalization)
# ---------------------------------------
def safe_norm(x):
    x = np.array(x, dtype=float)
    mx = x.max()
    mn = x.min()
    if np.isfinite(mx) and mx > mn:
        return (x - mn) / (mx - mn)
    elif mx == mn:
        return np.zeros_like(x)
    else:
        # fallback: clip NaNs
        return np.nan_to_num(x)

lc_n = safe_norm(least_conf)
loss_n = safe_norm(losses)
ann_n = safe_norm(annotator_disagreement)
cl_n = safe_norm(cleanlab_score)

active_score = (
    0.40 * lc_n +
    0.30 * loss_n +
    0.20 * ann_n +
    0.10 * cl_n
)

# ---------------------------------------
# Create a DataFrame (index-compatible)
# ---------------------------------------
active_df = pd.DataFrame({
    "sample_id": np.arange(len(y_series)),
    "true_label": y_series.values,
    "least_conf": least_conf,
    "entropy": entropy_vals,
    "loss": losses,
    "annotator_disagreement": annotator_disagreement,
    "cleanlab_score": cleanlab_score,
    "active_score": active_score
})

# ---------------------------------------
# 5) TOP-1% SAMPLE CHOOSING
# ---------------------------------------
TOP_RATIO = 0.01  # %1
top_k = max(1, int(len(active_df) * TOP_RATIO))

human_review = active_df.sort_values("active_score", ascending=False).head(top_k)

print("\n=== Samples Selected for Human Review ===")
print(human_review[["sample_id", "active_score", "least_conf", "loss"]])

# ---------------------------------------
# 6) SIMULATE HUMAN ANNOTATION (if needed)
# ---------------------------------------
np.random.seed(42)
new_labels_sim = []
for _, row in human_review.iterrows():
    old_label = row["true_label"]
    if np.random.rand() < 0.85:
        new_labels_sim.append(old_label)  # %85 doğru
    else:
        choices = [c for c in model.classes_ if c != old_label]
        if len(choices) == 0:
            new_labels_sim.append(old_label)
        else:
            new_labels_sim.append(np.random.choice(choices))  # %15 yanlış

# ---------------------------------------
# 7) LABEL UPDATE (safe updates on lineage_df and y_series)
# ---------------------------------------
# Ensure lineage_df has revision_id and annotation_timestamp columns
if "revision_id" not in lineage_df.columns:
    lineage_df["revision_id"] = 0
if "annotation_timestamp" not in lineage_df.columns:
    lineage_df["annotation_timestamp"] = pd.NaT

# Apply updates using sample_id indices (human_review.sample_id holds sample indices)
for idx, corrected_label in zip(human_review["sample_id"].astype(int).values, new_labels_sim):
    # Update y_series
    y_series.iloc[idx] = corrected_label
    # Update lineage_df: if there's an entry for this sample_id, update first occurrence; otherwise append
    rows = lineage_df.index[lineage_df.get("sample_id", pd.Series(np.arange(len(lineage_df)))) == idx].tolist()
    if len(rows) > 0:
        r = rows[0]
        lineage_df.at[r, "true_label"] = corrected_label
        lineage_df.at[r, "revision_id"] = int(lineage_df.at[r, "revision_id"]) + 1
        lineage_df.at[r, "annotation_timestamp"] = pd.Timestamp.now()
    else:
        # append a new row
        new_row = {
            "sample_id": int(idx),
            "annotator_id": "human_reviewer",
            "true_label": corrected_label,
            "revision_id": 1,
            "annotation_timestamp": pd.Timestamp.now()
        }
        lineage_df = pd.concat([lineage_df, pd.DataFrame([new_row])], ignore_index=True)

# Persist updates back to original variable names to keep pipeline consistent
# If original variable was numpy array, update it too
if isinstance(y_train_final, np.ndarray):
    # convert back to numpy
    y_train_final = y_series.values
else:
    # update the original Series object
    y_train_final = y_series

# If global names exist, update them
if "lineage_train" in globals():
    lineage_train = lineage_df
if "train_lineage" in globals():
    train_lineage = lineage_df

print(f"\nActive learning loop completed — {top_k} sample(s) updated.")
print("Modeli yeniden eğitmek için: model.fit(X_train_selected, y_train_final)")



# =========================
# G) NOISE REPAIR (AUTO-CORRECTION)
# =========================
import numpy as np
import pandas as pd
import time
from datetime import datetime
from sklearn.metrics import accuracy_score
from copy import deepcopy
import json

# ---------- Safety checks (robust) ----------
required_any = ["X_train_selected", "y_train_final", "X_test_selected", "y_test"]
missing_any = [name for name in required_any if name not in globals()]
if missing_any:
    raise RuntimeError(f"Missing required objects in environment: {missing_any} - please prepare them before running repair.")

# Determine lineage table variable safely (prefer train_lineage, then lineage_train)
if "train_lineage" in globals():
    lineage_df = train_lineage
elif "lineage_train" in globals():
    lineage_df = lineage_train
else:
    raise RuntimeError("Neither 'train_lineage' nor 'lineage_train' exists. Please provide lineage table before running repair.")

# Make copies / safe types
X_train_sel = X_train_selected.copy().reset_index(drop=True)
X_test_sel = X_test_selected.copy().reset_index(drop=True)

# y_train_final may be numpy array or pd.Series
y_was_numpy = isinstance(y_train_final, np.ndarray)
if y_was_numpy:
    y_series = pd.Series(y_train_final).reset_index(drop=True)
else:
    y_series = pd.Series(y_train_final).reset_index(drop=True)

# ---------- Configurable thresholds ----------
NOISE_CANDIDATE_THRESHOLD = 0.6   # final_noise_score above -> candidate for repair
HARD_RELABEL_PROB = 0.95          # model must be >= this to hard relabel
SOFT_ALPHA = 0.7                  # weighting for soft relabel: alpha * model_proba + (1-alpha) * one_hot(original)
MULTI_TOPK = 3                    # keep top-K probabilities for multi-label merging

# ---------- Resolve model(s) / ensemble ----------
ensemble_models = None
if "models" in globals() and isinstance(models, (list, tuple)) and len(models) > 0:
    ensemble_models = models
elif "model" in globals():
    ensemble_models = [model]
else:
    raise RuntimeError("No model(s) found. Ensure `models` or `model` exist in the environment.")

# ---------- Build unified classes (use labels present in training data) ----------
unified_classes = list(pd.Series(y_series).unique())
# ensure deterministic ordering (optional)
unified_classes = list(unified_classes)

# helper to align a single model's predict_proba output to unified class order
def align_proba_to_unified(model, proba):
    """
    model.classes_ may be in different order; proba is (n_samples, n_model_classes).
    Return aligned array (n_samples, n_unified_classes) where columns correspond to unified_classes.
    """
    aligned = np.zeros((proba.shape[0], len(unified_classes)), dtype=float)
    model_classes = list(model.classes_)
    map_idx = {c: i for i, c in enumerate(model_classes)}
    for j, cls in enumerate(unified_classes):
        if cls in map_idx:
            aligned[:, j] = proba[:, map_idx[cls]]
        else:
            # if model doesn't know this class, leave zeros (will be averaged across ensemble)
            aligned[:, j] = 0.0
    return aligned

# ---------- Build ensemble probability matrix on train (aligned) ----------
def ensemble_proba_matrix(X):
    proba_list = []
    for m in ensemble_models:
        # fit quickly if not fitted (use current labels)
        if not hasattr(m, "classes_"):
            try:
                m.fit(X, y_series.values)
            except Exception as e:
                # if fit fails, skip this model
                print(f"Warning: model {m} fit failed: {e}. Skipping.")
                continue
        p = m.predict_proba(X)  # shape (n_samples, n_model_classes)
        p_aligned = align_proba_to_unified(m, p)  # shape (n_samples, n_unified)
        proba_list.append(p_aligned)
    if len(proba_list) == 0:
        raise RuntimeError("No ensemble model produced probabilities.")
    stacked = np.stack(proba_list, axis=0)  # (n_models, n_samples, n_unified)
    mean_proba = np.mean(stacked, axis=0)
    # ensure numerical stability: rows sum to 1 if possible
    row_sums = mean_proba.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    mean_proba = mean_proba / row_sums
    return mean_proba

print("Computing ensemble probabilities on training set...")
train_ensemble_proba = ensemble_proba_matrix(X_train_sel)  # shape (n_samples, n_unified_classes)

# ---------- Identify candidate noisy samples ----------
# prefer noise_df["final_noise_score"] if exists and shape matches, else fallback to 1-max_proba
if "noise_df" in globals() and "final_noise_score" in noise_df.columns and len(noise_df["final_noise_score"]) == len(y_series):
    scores_all = pd.Series(noise_df["final_noise_score"]).reset_index(drop=True).values
else:
    # fallback: uncertainty proxy
    scores_all = 1.0 - train_ensemble_proba.max(axis=1)

candidates_mask = scores_all > NOISE_CANDIDATE_THRESHOLD
candidate_indices = np.where(candidates_mask)[0]
print(f"Found {len(candidate_indices)} candidate samples (threshold {NOISE_CANDIDATE_THRESHOLD}).")

# ---------- Backup original labels ----------
y_train_before = y_series.copy()

# ---------- Prepare lineage updates collector ----------
if "revision_id" not in lineage_df.columns:
    lineage_df["revision_id"] = 0
if "annotation_timestamp" not in lineage_df.columns:
    lineage_df["annotation_timestamp"] = pd.NaT
if "sample_id" not in lineage_df.columns:
    lineage_df["sample_id"] = np.arange(len(lineage_df))

repairs = []

# ---------- 1) HARD RELABELING ----------
print("Running HARD relabeling step...")
hard_relabels = []
for idx in candidate_indices:
    orig_label = y_series.iloc[idx]
    proba = train_ensemble_proba[idx]  # aligned to unified_classes
    top_idx = int(np.argmax(proba))
    top_prob = float(proba[top_idx])
    pred_label = unified_classes[top_idx]
    if (top_prob >= HARD_RELABEL_PROB) and (pred_label != orig_label):
        # apply hard relabel
        y_series.iloc[idx] = pred_label
        # update lineage row(s) for this sample_id: update first matching row if exists
        rows = lineage_df.index[lineage_df["sample_id"] == idx].tolist()
        if len(rows) > 0:
            r = rows[0]
            lineage_df.at[r, "true_label"] = pred_label
            lineage_df.at[r, "revision_id"] = int(lineage_df.at[r, "revision_id"]) + 1
            lineage_df.at[r, "annotation_timestamp"] = pd.Timestamp.now()
        else:
            # append new row
            new_row = {
                "sample_id": int(idx),
                "true_label": pred_label,
                "annotator_id": "auto_relabel",
                "revision_id": 1,
                "annotation_timestamp": pd.Timestamp.now()
            }
            lineage_df = pd.concat([lineage_df, pd.DataFrame([new_row])], ignore_index=True)
        repairs.append({
            "sample_id": int(idx),
            "method": "hard_relabel",
            "original_label": orig_label,
            "new_label": pred_label,
            "confidence": top_prob,
            "timestamp": datetime.now().isoformat()
        })
        hard_relabels.append(idx)

print(f"Hard relabels applied: {len(hard_relabels)}")

# ---------- 2) SOFT RELABELING ----------
print("Running SOFT relabeling step...")
soft_updated = []
for idx in candidate_indices:
    if idx in hard_relabels:
        continue
    orig_label = y_train_before.iloc[idx]
    proba = train_ensemble_proba[idx]
    # build one-hot for original in unified ordering
    one_hot_orig = np.zeros_like(proba)
    try:
        orig_pos = unified_classes.index(orig_label)
        one_hot_orig[orig_pos] = 1.0
    except ValueError:
        # original label not in unified_classes (unlikely) — skip one_hot
        pass
    soft_target = SOFT_ALPHA * proba + (1.0 - SOFT_ALPHA) * one_hot_orig
    # store soft target in lineage as JSON
    lineage_df.at[lineage_df.index[lineage_df["sample_id"] == idx].tolist()[0] if (idx in lineage_df["sample_id"].values) else lineage_df.index.max() + 1, "soft_target"] = json.dumps({
        "classes": unified_classes,
        "probs": soft_target.tolist()
    })
    # if no existing row, ensure appending of the soft target row (handled below if needed)
    if idx in lineage_df["sample_id"].values:
        r = lineage_df.index[lineage_df["sample_id"] == idx].tolist()[0]
        lineage_df.at[r, "revision_id"] = int(lineage_df.at[r, "revision_id"]) + 1
        lineage_df.at[r, "annotation_timestamp"] = pd.Timestamp.now()
    else:
        new_row = {
            "sample_id": int(idx),
            "true_label": orig_label,
            "annotator_id": "auto_soft",
            "soft_target": json.dumps({"classes": unified_classes, "probs": soft_target.tolist()}),
            "revision_id": 1,
            "annotation_timestamp": pd.Timestamp.now()
        }
        lineage_df = pd.concat([lineage_df, pd.DataFrame([new_row])], ignore_index=True)
    repairs.append({
        "sample_id": int(idx),
        "method": "soft_relabel",
        "original_label": orig_label,
        "soft_target_top3": [(unified_classes[i], float(proba[i])) for i in np.argsort(proba)[-3:][::-1]],
        "timestamp": datetime.now().isoformat()
    })
    soft_updated.append(idx)

print(f"Soft relabels prepared (soft targets stored): {len(soft_updated)}")

# ---------- 3) PROBABILISTIC MULTI-LABEL MERGING ----------
print("Running PROBABILISTIC multi-label merging (storing top-K distributions)...")
multi_updated = []
for idx in candidate_indices:
    proba = train_ensemble_proba[idx]
    topk_idx = np.argsort(proba)[-MULTI_TOPK:][::-1]
    multi = [(unified_classes[i], float(proba[i])) for i in topk_idx]
    # update or append in lineage_df
    if idx in lineage_df["sample_id"].values:
        r = lineage_df.index[lineage_df["sample_id"] == idx].tolist()[0]
        lineage_df.at[r, "probabilistic_labels"] = json.dumps(multi)
        lineage_df.at[r, "revision_id"] = int(lineage_df.at[r, "revision_id"]) + 1
        lineage_df.at[r, "annotation_timestamp"] = pd.Timestamp.now()
    else:
        new_row = {
            "sample_id": int(idx),
            "annotator_id": "auto_prob_merge",
            "probabilistic_labels": json.dumps(multi),
            "revision_id": 1,
            "annotation_timestamp": pd.Timestamp.now()
        }
        lineage_df = pd.concat([lineage_df, pd.DataFrame([new_row])], ignore_index=True)
    repairs.append({
        "sample_id": int(idx),
        "method": "probabilistic_merge",
        "topk": multi,
        "timestamp": datetime.now().isoformat()
    })
    multi_updated.append(idx)

print(f"Probabilistic multi-label records written: {len(multi_updated)}")

# ---------- Save repair log artifact ----------
repairs_df = pd.DataFrame(repairs)
repairs_df.to_csv("repair_actions_log.csv", index=False)
print("Repair actions saved to repair_actions_log.csv")

# ---------- POST-REPAIR VALIDATION ----------
print("Running post-repair validation: retrain model and compare on test set...")

# retrain a fresh copy of the first ensemble model (safe deepcopy)
retrain_model = deepcopy(ensemble_models[0])
try:
    retrain_model.fit(X_train_sel, y_series.values)
except Exception as e:
    print("Retrain failed:", e)
    retrain_model = None

# compute before/after using first original model (if available)
try:
    y_test_pred_before = ensemble_models[0].predict(X_test_sel)
    acc_before = accuracy_score(y_test, y_test_pred_before)
except Exception:
    acc_before = float("nan")

if retrain_model is not None:
    try:
        y_test_pred_after = retrain_model.predict(X_test_sel)
        acc_after = accuracy_score(y_test, y_test_pred_after)
    except Exception:
        acc_after = float("nan")
else:
    acc_after = float("nan")

print(f"Test Accuracy BEFORE repair (model[0]): {acc_before:.4f}")
print(f"Test Accuracy AFTER repair  (retrained): {acc_after:.4f}")

# Save lineage table with repairs (use train_lineage name if exists else lineage_train)
if "train_lineage" in globals():
    train_lineage = lineage_df
if "lineage_train" in globals():
    lineage_train = lineage_df

lineage_df.to_csv("lineage_train_post_repair.csv", index=False)
repairs_df.to_csv("repair_actions_log.csv", index=False)

print("Lineage table saved to lineage_train_post_repair.csv")
print("G) Noise repair completed.")



# ===============================================================
# H) POST-REPAIR VALIDATION
# ===============================================================

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
import pandas as pd

print("\n================ POST-REPAIR VALIDATION ================\n")

# -----------------------------------------------------------
# Ensure we have a model variable; prefer 'model', else try 'models[0]'
# -----------------------------------------------------------
if "model" in globals():
    _model = model
elif "models" in globals() and isinstance(models, (list, tuple)) and len(models) > 0:
    _model = models[0]
else:
    raise RuntimeError("No model found in globals(). Define `model` or `models` before running validation.")

# Ensure X_test_selected and X_train_selected exist
if "X_test_selected" not in globals() or "X_train_selected" not in globals():
    raise RuntimeError("X_test_selected or X_train_selected not found in environment.")

# make local copies to avoid accidental mutation
X_test_sel = X_test_selected.copy().reset_index(drop=True)
X_train_sel = X_train_selected.copy().reset_index(drop=True)

# Ensure y_test exists
if "y_test" not in globals():
    raise RuntimeError("y_test not found in environment.")

y_test_local = np.asarray(y_test).ravel()

# -----------------------------------------------------------
# 1) BEFORE METRICS (Before Repair)
# -----------------------------------------------------------
if not ("before_acc" in globals() and "before_f1" in globals()):
    # Ensure model is fitted; if not, fit on current training labels (y_train_final)
    if not hasattr(_model, "classes_"):
        if "y_train_final" not in globals():
            raise RuntimeError("Model not fitted and y_train_final not available to fit a baseline model.")
        print("Baseline model not fitted — fitting baseline model with current training labels for BEFORE metrics...")
        # Use y_train_final as fallback (may be ndarray or Series)
        y_train_for_fit = np.asarray(y_train_final).ravel()
        _model.fit(X_train_sel, y_train_for_fit)

    # Predictions and metrics
    y_pred_before = _model.predict(X_test_sel)
    before_acc  = accuracy_score(y_test_local, y_pred_before)
    before_f1   = f1_score(y_test_local, y_pred_before, average="macro")
    before_cm   = confusion_matrix(y_test_local, y_pred_before)

    # Expose to globals so subsequent runs can reuse
    globals()["before_acc"] = before_acc
    globals()["before_f1"] = before_f1
    globals()["before_cm"] = before_cm

    print(f"BEFORE Accuracy: {before_acc:.4f}")
    print(f"BEFORE F1-macro: {before_f1:.4f}")
else:
    print("BEFORE metrics already computed — reusing values.")
    before_acc = globals()["before_acc"]
    before_f1  = globals()["before_f1"]
    before_cm  = globals().get("before_cm", None)

# -----------------------------------------------------------
# 2) MODEL RETRAIN (after repair)
# -----------------------------------------------------------

print("\nRetraining model on repaired labels...")

# Determine y_train_repaired: prefer variable if exists, else fallback to y_train_final
if "y_train_repaired" in globals():
    y_train_repaired_local = np.asarray(y_train_final).ravel()
else:
    # fallback and warn
    print("WARNING: y_train_repaired not found → using y_train_final as repaired labels (may be unchanged).")
    if "y_train_final" not in globals():
        raise RuntimeError("y_train_final not found to use as fallback for retraining.")
    y_train_repaired_local = np.asarray(y_train_final).ravel()

# make a fresh copy of model for retraining if we want to preserve original baseline
from copy import deepcopy
_retrain_model = deepcopy(_model)

# Fit retrain model
_retrain_model.fit(X_train_sel, y_train_repaired_local)

print("Model retrained on repaired labels.")

# -----------------------------------------------------------
# 3) AFTER METRICS (After Repair)
# -----------------------------------------------------------

y_pred_after = _retrain_model.predict(X_test_sel)

after_acc = accuracy_score(y_test_local, y_pred_after)
after_f1  = f1_score(y_test_local, y_pred_after, average="macro")
after_cm  = confusion_matrix(y_test_local, y_pred_after)

# expose to globals for later inspection if needed
globals()["after_acc"] = after_acc
globals()["after_f1"] = after_f1
globals()["after_cm"] = after_cm

print(f"\nAFTER Accuracy: {after_acc:.4f}")
print(f"AFTER F1-macro: {after_f1:.4f}")

# -----------------------------------------------------------
# 4) METRIC LIFT
# -----------------------------------------------------------

acc_lift = after_acc - before_acc
f1_lift  = after_f1  - before_f1

print("\n========== METRIC LIFT ==========")
print(f"Accuracy Lift: {acc_lift:+.4f}")
print(f"F1-macro Lift: {f1_lift:+.4f}")

# -----------------------------------------------------------
# 5) Stability / Regression Check
# -----------------------------------------------------------

print("\n========== STABILITY / REGRESSION CHECK ==========")

def check_regression(before, after, name, tolerance=0.01):
    """
    If the performance drop exceeds the tolerance, trigger an alarm.
    """
    drop = before - after
    if drop > tolerance:
        print(f"❌ REGRESSION DETECTED in {name}: drop={drop:.4f}")
        return False
    else:
        print(f"✅ {name} stable (drop={drop:.4f})")
        return True

stable_acc = check_regression(before_acc, after_acc, "Accuracy")
stable_f1  = check_regression(before_f1,  after_f1,  "F1-macro")

pipeline_stable = stable_acc and stable_f1

# -----------------------------------------------------------
# 6) Human-in-the-Loop Approval Report
# -----------------------------------------------------------

qa_report = {
    "before_accuracy": float(before_acc),
    "after_accuracy": float(after_acc),
    "accuracy_lift": float(acc_lift),
    
    "before_f1_macro": float(before_f1),
    "after_f1_macro": float(after_f1),
    "f1_lift": float(f1_lift),

    "regression_free": bool(pipeline_stable),
}

qa_report_df = pd.DataFrame([qa_report])
print("\n========== QA APPROVAL REPORT ==========")
print(qa_report_df)

print("\nPOST-REPAIR VALIDATION completed.\n")



# ===============================================================
# I) SEVERITY-WEIGHTED NOISE SCORING (Noise Priority Score - NPS)
# ===============================================================
print("\n================ I) SEVERITY-WEIGHTED NOISE SCORING ================\n")

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.spatial.distance import mahalanobis

# -----------------------------------------
# 0. Ensure labels are numeric (Iris: 0,1,2)
# -----------------------------------------
if y_train_final.dtype == "object":
    print("Converting categorical labels to numeric...")
    le = LabelEncoder()
    y_train_final = le.fit_transform(y_train_final)

# -----------------------------------------
# 1. Prediction & estimated loss
# -----------------------------------------
y_pred_proba = model.predict_proba(X_train_selected)

true_class_probs = y_pred_proba[np.arange(len(y_train_final)), y_train_final]
y_pred_train = np.argmax(y_pred_proba, axis=1)

sample_loss = -np.log(true_class_probs + 1e-9)

df_noise = pd.DataFrame()
df_noise["sample_id"] = np.arange(len(y_train_final))
df_noise["true_label"] = y_train_final
df_noise["pred_label"] = y_pred_train
df_noise["confidence"] = true_class_probs
df_noise["est_model_loss"] = sample_loss

# -----------------------------------------
# 2. Mahalanobis anomaly (sensor space)
# -----------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train_selected)

mean = np.mean(X_scaled, axis=0)
cov = np.cov(X_scaled, rowvar=False)
inv_cov = np.linalg.pinv(cov)

def mahalanobis_distance(x):
    return mahalanobis(x, mean, inv_cov)

df_noise["embedding_anomaly"] = np.apply_along_axis(mahalanobis_distance, 1, X_scaled)

# normalize
df_noise["embedding_anomaly"] = (
    (df_noise["embedding_anomaly"] - df_noise["embedding_anomaly"].min()) /
    (df_noise["embedding_anomaly"].max() - df_noise["embedding_anomaly"].min() + 1e-9)
)

# -----------------------------------------
# 3. Class distance
# -----------------------------------------
cm = confusion_matrix(y_train_final, y_pred_train)
cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-9)

df_noise["class_distance"] = [
    1 - cm_norm[int(t)][int(p)]
    for t, p in zip(df_noise["true_label"], df_noise["pred_label"])
]

# -----------------------------------------
# 4. Logical contradiction flag
# -----------------------------------------
df_noise["logical_flag"] = (df_noise["true_label"] != df_noise["pred_label"]).astype(int)

# -----------------------------------------
# 5. Feature vector magnitude instability
# -----------------------------------------
sensor_magnitude = np.linalg.norm(X_train_selected, axis=1)

df_noise["sensor_magnitude"] = (
    (sensor_magnitude - sensor_magnitude.min()) /
    (sensor_magnitude.max() - sensor_magnitude.min() + 1e-9)
)

# -----------------------------------------
# 6. Normalize other numerical metrics
# -----------------------------------------
for col in ["est_model_loss", "class_distance"]:
    df_noise[col] = (
        (df_noise[col] - df_noise[col].min()) /
        (df_noise[col].max() - df_noise[col].min() + 1e-9)
    )

# -----------------------------------------
# 7. FINAL NOISE PRIORITY SCORE
# -----------------------------------------
w1, w2, w3, w4, w5 = 0.30, 0.25, 0.20, 0.15, 0.10

df_noise["NPS"] = (
    w1 * df_noise["est_model_loss"] +
    w2 * df_noise["class_distance"] +
    w3 * df_noise["embedding_anomaly"] +
    w4 * (df_noise["logical_flag"] * 2) +
    w5 * df_noise["sensor_magnitude"]
)

df_noise = df_noise.sort_values("NPS", ascending=False)

print("\n✅ Noise Priority Score calculated successfully\n")

# -----------------------------------------
# 8. Top priority noisy samples
# -----------------------------------------
top_5_percent = int(len(df_noise) * 0.05)
priority_samples = df_noise.head(top_5_percent)

print("Top 10 highest-priority noisy samples:\n")
print(priority_samples[["sample_id", "true_label", "pred_label", "NPS"]].head(10))

priority_samples.to_csv("Noise_Priority_List_iris.csv", index=False)

print("\n✅ Noise Priority List saved as: Noise_Priority_List_iris.csv")
print("\nI) Severity-Weighted Noise Scoring COMPLETED.\n")



# ================================================================
# J) FULL BIAS AUDITING
# ================================================================
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder

print("\n=== START: FULL BIAS AUDIT (Iris Dataset) ===\n")

# -------------------------
# 0) CHECK REQUIRED OBJECTS
# -------------------------
_required = ["model", "X_train_selected", "X_test_selected", "y_train_final", "y_test"]
_missing = [o for o in _required if o not in globals()]
if _missing:
    raise NameError(f"Required objects missing from workspace: {_missing}\nPlease define these before running the audit.")

# grab globals (work on local copies)
_model = globals()["model"]
X_train = globals()["X_train_selected"].copy()
X_test  = globals()["X_test_selected"].copy()
y_train_raw = globals()["y_train_final"]
y_test_raw  = globals()["y_test"]

# ensure shapes consistent
if len(y_test_raw) != len(X_test):
    raise ValueError(f"Length mismatch: len(y_test)={len(y_test_raw)} vs len(X_test)={len(X_test)}")

# output folder
OUT_DIR = "bias_audit_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------
# Helper: normalize label to string
# -------------------------
def normalize_label(x):
    """Return stable string for label x (handles bytes, pd.NA, numpy scalars)."""
    try:
        if pd.isna(x):
            return "__MISSING__"
    except Exception:
        pass
    if isinstance(x, (bytes, bytearray)):
        try:
            return x.decode()
        except Exception:
            return str(x)
    if isinstance(x, (np.integer,)):
        return str(int(x))
    if isinstance(x, (np.floating,)):
        return str(float(x))
    return str(x)

# -------------------------
# 1) NORMALIZE / ENCODE LABELS (fit LabelEncoder on union)
# -------------------------
y_test_norm = np.array([normalize_label(v) for v in np.asarray(y_test_raw)], dtype=object)
y_train_norm = np.array([normalize_label(v) for v in np.asarray(y_train_raw)], dtype=object)

union_labels = np.concatenate([y_train_norm, y_test_norm])
le = LabelEncoder().fit(union_labels)

y_train_enc = le.transform(y_train_norm)
y_test_enc  = le.transform(y_test_norm)

# Save label mapping (encoder classes are strings)
label_mapping = {int(i): label for i, label in enumerate(le.classes_)}
with open(os.path.join(OUT_DIR, "label_mapping.json"), "w") as f:
    json.dump(label_mapping, f, indent=2)

print("Label encoder classes:", le.classes_)

# -------------------------
# 2) PREDICTIONS + PROBABILITIES (aligned)
# -------------------------
has_proba = hasattr(_model, "predict_proba")
if has_proba:
    y_proba_test = _model.predict_proba(X_test)  # (N, C_model)
    # Map model.classes_ to encoder classes if needed
    # Convert model.classes_ to normalized-string domain for safe mapping
    model_classes_norm = np.array([normalize_label(c) for c in getattr(_model, "classes_", [])], dtype=object)
    # Build mapping from model class index -> encoder index
    model_to_encoder_idx = []
    if len(model_classes_norm) > 0:
        for mc in model_classes_norm:
            if mc in le.classes_:
                model_to_encoder_idx.append(int(np.where(le.classes_ == mc)[0][0]))
            else:
                model_to_encoder_idx.append(None)
    else:
        model_to_encoder_idx = [None] * y_proba_test.shape[1]

    # Align proba into encoder order (N, num_encoder_labels)
    proba_aligned = np.zeros((y_proba_test.shape[0], len(le.classes_)), dtype=float)
    for j, idx in enumerate(model_to_encoder_idx):
        if idx is not None:
            proba_aligned[:, idx] = y_proba_test[:, j]
    # if some encoder classes have no probability mass from model, leave zeros
    # renormalize rows (avoid div by zero)
    row_sums = proba_aligned.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    proba_aligned = proba_aligned / row_sums
    y_pred_test_enc = np.argmax(proba_aligned, axis=1)
    y_proba_test_aligned = proba_aligned
else:
    # use predict output and normalize
    y_pred_test_raw = _model.predict(X_test)
    y_pred_test_norm = np.array([normalize_label(v) for v in y_pred_test_raw], dtype=object)
    y_pred_test_enc = le.transform(y_pred_test_norm)
    y_proba_test_aligned = None

# convert to ints
y_test = np.asarray(y_test_enc, dtype=int)
y_pred = np.asarray(y_pred_test_enc, dtype=int)

print("Predictions computed. Has predict_proba:", has_proba)

# -------------------------
# 3) BASIC OVERALL METRICS
# -------------------------
labels_sorted = np.unique(np.concatenate([y_test, y_pred]))

overall_acc = accuracy_score(y_test, y_pred)
prec, rec, f1, support = precision_recall_fscore_support(
    y_test, y_pred, average=None, labels=labels_sorted, zero_division=0
)
class_report = classification_report(y_test, y_pred, digits=4)

print("Overall Accuracy: {:.4f}".format(overall_acc))
print("\nClassification report (per class):\n")
print(class_report)

# save classification report text
with open(os.path.join(OUT_DIR, "classification_report.txt"), "w") as f:
    f.write(f"Overall Accuracy: {overall_acc:.4f}\n\n")
    f.write(class_report)

# -------------------------
# 4) CONFUSION MATRIX (visual + csv)
# -------------------------
cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)

# axis labels in human-readable form
axis_labels = [label_mapping[int(l)] for l in labels_sorted]

plt.figure(figsize=(7,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=axis_labels, yticklabels=axis_labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (Iris Dataset)")
cm_path = os.path.join(OUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path, bbox_inches="tight")
plt.close()
print(f"Confusion matrix saved -> {cm_path}")

cm_df = pd.DataFrame(cm, index=axis_labels, columns=axis_labels)
cm_df.to_csv(os.path.join(OUT_DIR, "confusion_matrix.csv"))

# -------------------------
# 5) SUBJECT / CLASS BIAS CHECK (Per-class accuracy)
# -------------------------
unique_subjects = np.unique(y_test)
sub_accs = {}
sub_counts = {}
for s in unique_subjects:
    mask = (y_test == s)
    n = int(mask.sum())
    acc = accuracy_score(y_test[mask], y_pred[mask]) if n > 0 else np.nan
    sub_accs[int(s)] = float(acc) if not np.isnan(acc) else np.nan
    sub_counts[int(s)] = int(n)

sub_acc_df = pd.DataFrame({
    "subject_id_enc": list(sub_accs.keys()),
    "accuracy": list(sub_accs.values()),
    "n_samples": list(sub_counts.values())
})
sub_acc_df["subject_label"] = sub_acc_df["subject_id_enc"].map(lambda x: label_mapping[int(x)])
sub_acc_df.to_csv(os.path.join(OUT_DIR, "per_class_accuracy.csv"), index=False)

# plot
plt.figure(figsize=(8,4))
plt.bar(sub_acc_df["subject_label"], sub_acc_df["accuracy"])
plt.xlabel("Class (Species)")
plt.ylabel("Accuracy")
plt.title("Per-Class Accuracy (Iris Dataset)")
plt.xticks(rotation=45)
plt.grid(alpha=0.3)
plt.tight_layout()
sub_plot_path = os.path.join(OUT_DIR, "per_class_accuracy.png")
plt.savefig(sub_plot_path)
plt.close()
print(f"Per-class accuracy saved -> {sub_plot_path}")

sub_std = sub_acc_df["accuracy"].std()
sub_mean = sub_acc_df["accuracy"].mean()

subject_bias_risk = "LOW"
if sub_std > 0.08:
    subject_bias_risk = "HIGH"
elif sub_std > 0.04:
    subject_bias_risk = "MEDIUM"

print(f"Per-class accuracy mean={sub_mean:.4f}, std={sub_std:.4f} → risk={subject_bias_risk}")

# -------------------------
# 6) TIME-DRIFT
# -------------------------
# Not applicable for Iris; note in report
time_drift_note = "Not applicable (Iris dataset has no time split)."

# -------------------------
# 7) CLASS-LEVEL SKEW & PERFORMANCE
# -------------------------
class_counts = pd.Series(y_train_enc).value_counts().sort_index()
class_perf = []
for cls in labels_sorted:
    mask = (y_test == cls)
    n = int(mask.sum())
    acc_cls = accuracy_score(y_test[mask], y_pred[mask]) if n > 0 else np.nan
    class_perf.append({"class_enc": int(cls), "class_label": label_mapping[int(cls)], "n_test": n, "accuracy": float(acc_cls) if not np.isnan(acc_cls) else np.nan})
class_perf_df = pd.DataFrame(class_perf).sort_values("class_enc")
class_perf_df.to_csv(os.path.join(OUT_DIR, "class_level_performance.csv"), index=False)

plt.figure(figsize=(8,4))
plt.bar(class_perf_df["class_label"], class_perf_df["accuracy"])
plt.xlabel("Class (Species)")
plt.ylabel("Accuracy")
plt.title("Per-Class Accuracy")
plt.tight_layout()
cls_plot_path = os.path.join(OUT_DIR, "per_class_accuracy_summary.png")
plt.savefig(cls_plot_path)
plt.close()
print(f"Per-class accuracy summary saved -> {cls_plot_path}")

pd.DataFrame({"class_enc": class_counts.index.astype(int), "train_count": class_counts.values}).to_csv(os.path.join(OUT_DIR, "train_class_distribution.csv"), index=False)

# -------------------------
# 8) ANNOTATOR BIAS CHECK (not applicable)
# -------------------------
annotator_bias = {
    "exists": False,
    "reason": "Iris dataset does not have per-sample annotator_id metadata.",
    "recommendation": "If annotator metadata becomes available, compute per-annotator JS divergence vs global label distribution and confusion matrices."
}

# -------------------------
# 9) SENSOR-LEVEL FEATURE CHECKS (Iris heuristics)
# -------------------------
try:
    feature_magnitude = np.linalg.norm(X_test, axis=1)
    mag_df = pd.DataFrame({"mag": feature_magnitude, "true": y_test, "pred": y_pred})
    agg = mag_df.groupby("true")["mag"].agg(["mean","std","count"]).reset_index()
    agg["label"] = agg["true"].map(lambda x: label_mapping[int(x)])
    agg.to_csv(os.path.join(OUT_DIR, "feature_magnitude_by_true_class.csv"), index=False)
    sensor_bias_notes = agg.to_dict(orient="records")
except Exception as e:
    sensor_bias_notes = [{"error": str(e)}]

# -------------------------
# 10) NEAR/FAR CONFUSION METRIC
# -------------------------
cm_sum_rows = cm.sum(axis=1, keepdims=True)
cm_norm = cm / (cm_sum_rows + 1e-9)
label_to_idx = {int(l): i for i, l in enumerate(labels_sorted)}
near_far = []
for t, p in zip(y_test, y_pred):
    i = label_to_idx[int(t)]
    j = label_to_idx[int(p)]
    conf_prob = float(cm_norm[i, j])
    near_far.append(conf_prob)
np_test_nearfar = np.array(near_far)
median_conf_by_class = pd.DataFrame({"class_enc": labels_sorted, "median_conf": np.median(cm_norm, axis=1)})
median_conf_by_class["label"] = median_conf_by_class["class_enc"].map(lambda x: label_mapping[int(x)])
median_conf_by_class.to_csv(os.path.join(OUT_DIR, "median_conf_by_class.csv"), index=False)

# -------------------------
# 11) AGGREGATE REPORT OBJECT
# -------------------------
report = {
    "dataset": "Iris Dataset",
    "overall_accuracy": float(overall_acc),
    "per_class_mean_accuracy": float(sub_mean),
    "per_class_std_accuracy": float(sub_std),
    "per_class_bias_risk": subject_bias_risk,
    "class_counts_train": class_counts.to_dict(),
    "annotator_bias": annotator_bias,
    "sensor_bias_notes": sensor_bias_notes,
    "time_drift": time_drift_note,
    "notes": "Demographic bias (gender/age) not applicable in this dataset."
}

with open(os.path.join(OUT_DIR, "bias_audit_summary.json"), "w") as f:
    json.dump(report, f, indent=2)

summary_row = {
    "overall_accuracy": overall_acc,
    "per_class_mean_accuracy": sub_mean,
    "per_class_std_accuracy": sub_std,
    "per_class_bias_risk": subject_bias_risk
}
pd.DataFrame([summary_row]).to_csv(os.path.join(OUT_DIR, "bias_audit_report.csv"), index=False)

# -------------------------
# 12) EXECUTIVE SUMMARY (console)
# -------------------------
print("\n=== EXECUTIVE SUMMARY ===")
print(f"Dataset: Iris Dataset")
print(f"Overall accuracy: {overall_acc:.4f}")
print(f"Per-class accuracy: mean={sub_mean:.4f}, std={sub_std:.4f} -> risk={subject_bias_risk}")
print("Top per-class accuracies (class : accuracy):")
print(class_perf_df[["class_label", "accuracy"]].to_string(index=False))
print("\nNotes:")
print("- Demographic bias: Not applicable (no gender/age in dataset).")
print("- Annotator bias: Not available (no annotator IDs).")
print(f"- Feature-level notes (saved): {os.path.join(OUT_DIR, 'feature_magnitude_by_true_class.csv')}")
print("\nSaved all outputs to folder:", OUT_DIR)
print("\n=== END: FULL BIAS AUDIT ===\n")



# -----------------------
# K) MODEL ROBUSTIFICATION
# -----------------------
import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, f1_score

OUT_DIR = "robustification_outputs"
os.makedirs(OUT_DIR, exist_ok=True)
RANDOM_STATE = 42

# -----------------------
# 0) Obtain preprocessed data
# -----------------------
try:
    X_train_in = X_train_corr.copy()
    X_test_in  = X_test_corr.copy()
    y_train_in = pd.Series(y_train_final).reset_index(drop=True).copy()
    y_test_in  = pd.Series(y_test).reset_index(drop=True).copy()
    print("Using preprocessed datasets from pipeline: X_train_corr, X_test_corr, y_train_final, y_test")
except Exception as e:
    print("Warning: preprocessed datasets not found.", e)
    from sklearn.datasets import load_iris
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=[f"f{i}" for i in range(iris.data.shape[1])])
    y = pd.Series(iris.target)
    from sklearn.model_selection import train_test_split
    X_train_in, X_test_in, y_train_in, y_test_in = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

# Ensure DataFrame types
if not isinstance(X_train_in, pd.DataFrame):
    X_train_in = pd.DataFrame(X_train_in)
if not isinstance(X_test_in, pd.DataFrame):
    X_test_in = pd.DataFrame(X_test_in)

print("Train shape (pre-robust):", X_train_in.shape)
print("Test  shape (pre-robust):", X_test_in.shape)

# -----------------------
# 1) SAFE LABEL HANDLING (detect & fix mismatched label universes)
# -----------------------
# Normalize types (handle possible strings/objects)
y_train_raw = pd.Series(y_train_in).reset_index(drop=True)
y_test_raw  = pd.Series(y_test_in).reset_index(drop=True)

# Try to coerce to ints where possible, but keep originals for fallback
def try_int_series(s):
    try:
        return s.astype(int)
    except Exception:
        # try mapping unique strings to ints deterministically
        uniques = pd.Series(s.unique())
        mapping = {u: i for i, u in enumerate(sorted(uniques.astype(str)))}
        return s.astype(str).map(mapping).astype(int)

y_train_try = try_int_series(y_train_raw)
y_test_try  = try_int_series(y_test_raw)

train_uni = np.unique(y_train_try)
test_uni  = np.unique(y_test_try)

print("Unique labels (train):", train_uni)
print("Unique labels (test) :", test_uni)

# If universes are identical -> use them as-is
if set(train_uni.tolist()) == set(test_uni.tolist()):
    print("Label universes match -> no mapping required.")
    y_train_enc = y_train_try.values.astype(int)
    y_test_enc  = y_test_try.values.astype(int)
    applied_mapping = None
else:
    # If disjoint or different, attempt a safe positional mapping:
    # map sorted(test_uni) -> sorted(train_uni)
    if len(train_uni) != len(test_uni):
        print("Warning: different number of unique labels between train and test.")
    sorted_train = sorted(train_uni.tolist())
    sorted_test  = sorted(test_uni.tolist())

    # Build mapping from test_val -> train_val by positional alignment
    # e.g. [3,4,5] -> [0,1,2]
    applied_mapping = {}
    for i, tv in enumerate(sorted_test):
        tgt = sorted_train[i] if i < len(sorted_train) else sorted_train[-1]
        applied_mapping[int(tv)] = int(tgt)

    print("Applying automatic label mapping (test -> train):", applied_mapping)

    # Apply mapping to y_test_try (safe)
    y_test_mapped = y_test_try.map(lambda v: applied_mapping.get(int(v), int(v))).astype(int)

    # Final encodings
    y_train_enc = y_train_try.values.astype(int)
    y_test_enc  = y_test_mapped.values.astype(int)

# Save mapping info for auditing
with open(os.path.join(OUT_DIR, "label_universe_mapping.json"), "w") as f:
    json.dump({
        "train_unique": train_uni.tolist(),
        "test_unique_before": test_uni.tolist(),
        "applied_mapping_test_to_train": applied_mapping if 'applied_mapping' in locals() else None
    }, f, indent=2)

# save class info (train)
unique_classes = sorted(np.unique(y_train_enc))
with open(os.path.join(OUT_DIR, "robust_label_classes.json"), "w") as f:
    json.dump({"classes": [int(c) for c in unique_classes]}, f, indent=2)

# -----------------------
# 2) NORMALIZATION
# -----------------------
scaler = StandardScaler()
Xtr_norm = scaler.fit_transform(X_train_in.values)
Xte_norm = scaler.transform(X_test_in.values)

Xtr_norm_orig = Xtr_norm.copy()
y_train_orig = y_train_enc.copy()

print("✅ Normalization done.")

# -----------------------
# 3) AUGMENTATION
# -----------------------
def rotation_augment(X, y, noise_std=0.02, random_state=None):
    rng = np.random.RandomState(random_state)
    noise = rng.normal(0.0, noise_std, X.shape)
    return np.vstack((X, X + noise)), np.hstack((y, y))

Xtr_aug, ytr_aug = rotation_augment(Xtr_norm, y_train_enc, 0.02, RANDOM_STATE)
print("✅ Augmented:", Xtr_aug.shape)

# -----------------------
# 4) ORIENTATION-INVARIANT
# -----------------------
Xtr_inv = np.abs(Xtr_aug)
Xte_inv = np.abs(Xte_norm)

print("✅ Orientation-invariant applied")

# -----------------------
# 5) HARD SAMPLE DETECTION
# -----------------------
baseline = RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE)
baseline.fit(Xtr_inv, ytr_aug)

probs = baseline.predict_proba(Xtr_inv)
true_probs = probs[np.arange(len(probs)), ytr_aug]
loss = -np.log(true_probs + 1e-9)

threshold = np.percentile(loss, 85)
hard_idx = loss >= threshold
print("Unstable samples detected:", int(np.sum(hard_idx)), "/", len(loss))

# -----------------------
# 6) REWEIGHTING
# -----------------------
base_weights = compute_sample_weight("balanced", ytr_aug)
hard_boost = np.where(hard_idx, 3.0, 1.0)
sample_weights = base_weights * hard_boost

print("✅ Reweighting finished")

# -----------------------
# 7) FINAL ROBUST MODEL
# -----------------------
robust_model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced_subsample",
    random_state=RANDOM_STATE
)
robust_model.fit(Xtr_inv, ytr_aug, sample_weight=sample_weights)
print("✅ Robust model trained")

# -----------------------
# 8) EVALUATION
# -----------------------
y_pred_robust = robust_model.predict(Xte_inv)

robust_acc = accuracy_score(y_test_enc, y_pred_robust)
robust_f1  = f1_score(y_test_enc, y_pred_robust, average="macro")

print("\n========== ROBUST RESULTS ==========")
print("Accuracy :", round(robust_acc, 4))
print("F1 Macro :", round(robust_f1, 4))

pd.DataFrame({
    "metric": ["accuracy", "f1_macro"],
    "value": [robust_acc, robust_f1]
}).to_csv(os.path.join(OUT_DIR, "robust_metrics_iris.csv"), index=False)

# -----------------------
# 9) EXPORT GLOBALS
# -----------------------
X_train_selected = pd.DataFrame(Xtr_inv, columns=[f"f{i}" for i in range(Xtr_inv.shape[1])])
X_test_selected  = pd.DataFrame(Xte_inv, columns=[f"f{i}" for i in range(Xte_inv.shape[1])])
y_train_final = ytr_aug.copy()
y_test = y_test_enc.copy()

manifest = {
    "X_train_selected_shape": X_train_selected.shape,
    "y_train_final_len": len(y_train_final),
    "X_test_selected_shape": X_test_selected.shape,
    "y_test_len": len(y_test),
    "applied_mapping": applied_mapping if 'applied_mapping' in locals() else None
}
with open(os.path.join(OUT_DIR, "export_manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

print("\n✅ K) ROBUSTIFICATION COMPLETED")
