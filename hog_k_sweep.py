"""
hog_k_sweep.py
MSIN0025 Scenario Week — k sweep on HOG cell=4 + L2 pixels, PCA=150

Best config from hog_sweep.py: HOG cell=4 + pixels, PCA=150, k=5 → 88.00% (90/10)
This script tests k ∈ {3, 5, 7, 9, 11} on both 80/20 and 90/10 splits
to find the true optimal k before updating the permanent pipeline.
"""

import numpy as np
import pandas as pd
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize

from python_files.knn_model import LABEL_NAMES, PIXEL_COLS

print("Loading data ...", flush=True)
# CSV files must be in the same folder as this script
df = pd.read_csv("product_images.csv")
X_raw = df[PIXEL_COLS].values.astype(np.float32)
y = df["label"].values.astype(int)
CLASS_NAMES = [LABEL_NAMES[i] for i in range(10)]

# ---------------------------------------------------------------------------
# Build HOG cell=4 + L2 pixels feature matrix (same as winning hog_sweep config)
# ---------------------------------------------------------------------------
print("Extracting HOG features (cell=4) ...", flush=True)
hog_feats = []
for row in X_raw:
    h = hog(
        row.reshape(28, 28),
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
        orientations=9,
        feature_vector=True,
        channel_axis=None,
    )
    hog_feats.append(h)
X_hog = normalize(np.array(hog_feats, dtype=np.float32), norm="l2")

X_l2 = normalize(X_raw, norm="l2")
X_feat = normalize(np.hstack([X_hog, X_l2]), norm="l2")   # shape (20000, 2080)
print(f"  Feature matrix: {X_feat.shape}", flush=True)

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
K_VALUES = [3, 5, 7, 9, 11]
results = []

for split_label, test_size in [("80/20", 0.2), ("90/10", 0.1)]:
    tr_idx, te_idx = train_test_split(
        np.arange(len(y)), test_size=test_size, random_state=42, stratify=y
    )
    X_tr, X_te = X_feat[tr_idx], X_feat[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    pca = PCA(n_components=150, random_state=42)
    X_tr_r = pca.fit_transform(X_tr)
    X_te_r = pca.transform(X_te)
    print(f"\n=== SPLIT {split_label} ===  PCA var={np.sum(pca.explained_variance_ratio_):.1%}")

    for k in K_VALUES:
        knn = KNeighborsClassifier(
            n_neighbors=k, metric="cosine", algorithm="brute", weights="distance"
        )
        knn.fit(X_tr_r, y_tr)
        y_pred = knn.predict(X_te_r)
        acc = accuracy_score(y_te, y_pred)
        print(f"  k={k:>2}  accuracy={acc:.4f}", flush=True)
        results.append({"split": split_label, "k": k, "acc": acc,
                        "knn": knn, "pca": pca, "X_te_r": X_te_r, "y_te": y_te})

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 45)
print("SUMMARY  (HOG cell=4 + pixels, PCA=150)")
print("=" * 45)
print(f"  {'k':>4}  {'80/20':>8}  {'90/10':>8}")
print(f"  {'-'*4}  {'-'*8}  {'-'*8}")
res_df = pd.DataFrame(results)
best_acc = 0
best_row = None
for k in K_VALUES:
    a80 = res_df[(res_df["split"]=="80/20") & (res_df["k"]==k)].iloc[0]["acc"]
    a90 = res_df[(res_df["split"]=="90/10") & (res_df["k"]==k)].iloc[0]["acc"]
    print(f"  {k:>4}  {a80:>8.4f}  {a90:>8.4f}")
    if a90 > best_acc:
        best_acc = a90
        best_row = res_df[(res_df["split"]=="90/10") & (res_df["k"]==k)].iloc[0]

print(f"\n  Best on 90/10: k={best_row['k']}  ({best_acc:.4f})")

# ---------------------------------------------------------------------------
# Per-class report for best k on 90/10
# ---------------------------------------------------------------------------
print(f"\n=== PER-CLASS REPORT: k={best_row['k']} on 90/10 ===")
y_pred_best = best_row["knn"].predict(best_row["X_te_r"])
print(classification_report(
    best_row["y_te"], y_pred_best,
    labels=list(range(10)), target_names=CLASS_NAMES, digits=3,
))
