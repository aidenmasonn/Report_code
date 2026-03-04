"""
hog_sweep.py
MSIN0025 Scenario Week — HOG Feature Extraction + k-NN Experiment

HOG (Histogram of Oriented Gradients) captures local edge directions rather
than raw pixel intensities. For clothing images, this encodes structural
information: collar shapes, sleeve boundaries, strap geometry — exactly the
features that distinguish Shirt from T-shirt or Dress from Coat.

Pipeline: Raw pixels → HOG features → PCA → cosine k-NN (distance weights)

Sweeps:
  - HOG cell_size ∈ {4, 7} (4px cells = finer, 7px cells = coarser)
  - PCA n_components ∈ {50, 100, 125, 150}
  - k ∈ {3, 5, 7, 9}
  - Both 80/20 and 90/10 splits

Also compares HOG-only vs HOG + raw pixels (concatenated) to check whether
adding pixel intensities back in helps or hurts.
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
X_raw = df[PIXEL_COLS].values.astype(np.float32)   # shape (20000, 784)
y = df["label"].values.astype(int)

CLASS_NAMES = [LABEL_NAMES[i] for i in range(10)]

# ---------------------------------------------------------------------------
# HOG feature extraction
# ---------------------------------------------------------------------------
def extract_hog(X_pixels, pixels_per_cell=(4, 4), cells_per_block=(2, 2)):
    """
    Extract HOG features for every image in X_pixels.
    Each row is a flattened 28x28 image (784 values).
    Returns L2-normalised HOG feature matrix.
    """
    feats = []
    for row in X_pixels:
        img = row.reshape(28, 28)
        h = hog(
            img,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            orientations=9,         # 9 gradient direction bins (standard)
            feature_vector=True,
            channel_axis=None,      # grayscale
        )
        feats.append(h)
    feats = np.array(feats, dtype=np.float32)
    return normalize(feats, norm="l2")


# ---------------------------------------------------------------------------
# Helper: fit PCA + k-NN, return accuracy
# ---------------------------------------------------------------------------
def evaluate(X_tr, X_te, y_tr, y_te, n_components, k):
    pca = PCA(n_components=n_components, random_state=42)
    X_tr_r = pca.fit_transform(X_tr)
    X_te_r = pca.transform(X_te)
    knn = KNeighborsClassifier(
        n_neighbors=k,
        metric="cosine",
        algorithm="brute",
        weights="distance",
    )
    knn.fit(X_tr_r, y_tr)
    return accuracy_score(y_te, knn.predict(X_te_r)), knn, pca, X_te_r, y_te


# ---------------------------------------------------------------------------
# Extract HOG features for both cell sizes
# ---------------------------------------------------------------------------
print("Extracting HOG features (cell=4) ...", flush=True)
X_hog4 = extract_hog(X_raw, pixels_per_cell=(4, 4))
print(f"  HOG(cell=4) shape: {X_hog4.shape}")

print("Extracting HOG features (cell=7) ...", flush=True)
X_hog7 = extract_hog(X_raw, pixels_per_cell=(7, 7))
print(f"  HOG(cell=7) shape: {X_hog7.shape}")

# Also prepare L2-normalised raw pixels (current baseline)
X_l2 = normalize(X_raw, norm="l2")

# HOG + raw pixels concatenated (structural + intensity)
X_hog4_px = normalize(np.hstack([X_hog4, X_l2]), norm="l2")
print(f"  HOG(cell=4)+pixels shape: {X_hog4_px.shape}")

# ---------------------------------------------------------------------------
# Sweep configurations
# ---------------------------------------------------------------------------
CONFIGS = [
    ("Baseline (L2 pixels)",    X_l2,      125, 5),   # current best — reference
    ("HOG cell=4",              X_hog4,     50, 5),
    ("HOG cell=4",              X_hog4,    100, 5),
    ("HOG cell=4",              X_hog4,    125, 5),
    ("HOG cell=4",              X_hog4,    150, 5),
    ("HOG cell=4",              X_hog4,    125, 3),
    ("HOG cell=4",              X_hog4,    125, 7),
    ("HOG cell=7",              X_hog7,     50, 5),
    ("HOG cell=7",              X_hog7,    100, 5),
    ("HOG cell=7",              X_hog7,    125, 5),
    ("HOG cell=4 + pixels",     X_hog4_px, 125, 5),
    ("HOG cell=4 + pixels",     X_hog4_px, 150, 5),
]

# ---------------------------------------------------------------------------
# Run sweep on 80/20 and 90/10 splits
# ---------------------------------------------------------------------------
results = []
best_overall = {"acc": 0}

for split_label, test_size in [("80/20", 0.2), ("90/10", 0.1)]:
    X_tr, X_te, y_tr, y_te = train_test_split(
        np.zeros(len(y)),   # dummy — we'll replace per-config below
        y, test_size=test_size, random_state=42, stratify=y
    )
    tr_idx = np.where(np.isin(np.arange(len(y)), np.where(
        np.isin(np.arange(len(y)),
                train_test_split(np.arange(len(y)), test_size=test_size,
                                 random_state=42, stratify=y)[0]))[0]))[0]
    # Re-derive proper train/test indices once
    tr_idx, te_idx = train_test_split(
        np.arange(len(y)), test_size=test_size, random_state=42, stratify=y
    )
    y_tr, y_te = y[tr_idx], y[te_idx]

    print(f"\n=== SPLIT {split_label} ===", flush=True)
    for name, X_feat, n_pca, k in CONFIGS:
        X_feat_tr = X_feat[tr_idx]
        X_feat_te = X_feat[te_idx]
        acc, knn, pca, X_te_r, yte = evaluate(
            X_feat_tr, X_feat_te, y_tr, y_te, n_pca, k
        )
        tag = f"{name}  PCA={n_pca}  k={k}"
        print(f"  {tag:<45}  {split_label}: {acc:.4f}", flush=True)
        results.append({
            "split": split_label, "name": name,
            "pca": n_pca, "k": k, "acc": acc,
            "knn": knn, "pca_obj": pca, "X_te_r": X_te_r, "y_te": yte,
        })
        if acc > best_overall["acc"]:
            best_overall = {
                "acc": acc, "split": split_label, "tag": tag,
                "knn": knn, "X_te_r": X_te_r, "y_te": yte,
            }

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SUMMARY (sorted by 80/20 accuracy)")
print("=" * 70)
print(f"  {'Config':<45} {'80/20':>7} {'90/10':>7}")
print(f"  {'-'*45} {'-'*7} {'-'*7}")

res_df = pd.DataFrame(results)
for _, grp in res_df.groupby(["name", "pca", "k"]):
    row80 = grp[grp["split"] == "80/20"]
    row90 = grp[grp["split"] == "90/10"]
    if row80.empty or row90.empty:
        continue
    a80 = row80.iloc[0]["acc"]
    a90 = row90.iloc[0]["acc"]
    tag = f"{row80.iloc[0]['name']}  PCA={row80.iloc[0]['pca']}  k={row80.iloc[0]['k']}"
    print(f"  {tag:<45} {a80:>7.4f} {a90:>7.4f}")

# ---------------------------------------------------------------------------
# Per-class report for the best config found
# ---------------------------------------------------------------------------
print(f"\n=== PER-CLASS REPORT: best overall = {best_overall['tag']} ({best_overall['acc']:.4f}) ===")
y_pred_best = best_overall["knn"].predict(best_overall["X_te_r"])
print(classification_report(
    best_overall["y_te"], y_pred_best,
    labels=list(range(10)), target_names=CLASS_NAMES, digits=3,
))
