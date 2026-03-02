"""
k_sweep_current.py
MSIN0025 Scenario Week — k-Value Sweep at CURRENT Best Config

Tests k ∈ {1, 3, 5, 7, 9, 11, 15, 21, 31} with the current best config:
  - PCA = 125 components
  - metric = cosine
  - weights = distance
  - L2 normalisation applied (via load_data)

Uses BOTH an 80/20 and a 90/10 stratified split so we can see whether the
optimal k is stable across training-set sizes. Previous sweeps predated
L2 normalisation, so results may differ.

Also prints per-class F1 for the overall best k to diagnose Shirt (class 6).
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from python_files.knn_model import LABEL_NAMES, load_data

print("Loading data (L2-normalised) ...", flush=True)
X, y, X_pred, df, df_pred = load_data()

K_VALUES = [1, 3, 5, 7, 9, 11, 15, 21, 31]
CLASS_NAMES = [LABEL_NAMES[i] for i in range(10)]

# ---------------------------------------------------------------------------
# Helper: run sweep on a given train/test split
# ---------------------------------------------------------------------------
def run_sweep(X_train, X_test, y_train, y_test, label):
    pca = PCA(n_components=125, random_state=42)
    X_tr = pca.fit_transform(X_train)
    X_te = pca.transform(X_test)
    var = np.sum(pca.explained_variance_ratio_)
    print(f"\n  [{label}] train={X_tr.shape}  test={X_te.shape}  PCA var={var:.1%}")

    results = []
    for k in K_VALUES:
        knn = KNeighborsClassifier(
            n_neighbors=k,
            metric="cosine",
            algorithm="brute",
            weights="distance",
        )
        knn.fit(X_tr, y_train)
        y_pred = knn.predict(X_te)
        acc = accuracy_score(y_test, y_pred)
        results.append((k, acc, y_pred))
        print(f"    k={k:>2}  accuracy={acc:.4f}", flush=True)

    best_k, best_acc, best_pred = max(results, key=lambda r: r[1])
    return results, best_k, best_acc, best_pred, y_test

# ---------------------------------------------------------------------------
# 80/20 split
# ---------------------------------------------------------------------------
print("\n=== SPLIT 80/20 ===")
X_tr80, X_te80, y_tr80, y_te80 = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
res80, best_k80, best_acc80, best_pred80, yte80 = run_sweep(
    X_tr80, X_te80, y_tr80, y_te80, "80/20"
)

# ---------------------------------------------------------------------------
# 90/10 split (more training data — closer to final model conditions)
# ---------------------------------------------------------------------------
print("\n=== SPLIT 90/10 ===")
X_tr90, X_te90, y_tr90, y_te90 = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)
res90, best_k90, best_acc90, best_pred90, yte90 = run_sweep(
    X_tr90, X_te90, y_tr90, y_te90, "90/10"
)

# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------
print("\n" + "=" * 55)
print("SUMMARY")
print("=" * 55)
print(f"  {'k':>4}  {'80/20 acc':>10}  {'90/10 acc':>10}")
print(f"  {'-'*4}  {'-'*10}  {'-'*10}")
for (k, a80, _), (_, a90, _) in zip(res80, res90):
    m80 = "  ← 80 best" if k == best_k80 else ""
    m90 = "  ← 90 best" if k == best_k90 else ""
    marker = m80 or m90
    print(f"  {k:>4}  {a80:>10.4f}  {a90:>10.4f}{marker}")

# ---------------------------------------------------------------------------
# Per-class F1 for best k on 90/10 split (more representative)
# ---------------------------------------------------------------------------
print(f"\n=== PER-CLASS REPORT: best k={best_k90} on 90/10 split ===")
print(classification_report(
    yte90, best_pred90,
    labels=list(range(10)),
    target_names=CLASS_NAMES,
    digits=3,
))
