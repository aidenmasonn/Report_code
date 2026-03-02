"""
weight_comparison.py
MSIN0025 Scenario Week — Weighting Scheme Comparison

Benchmarks three neighbour-weighting formulas for k-NN (cosine, k=5, PCA=50):
  1. Uniform      — all 5 neighbours vote equally (pure majority)
  2. 1/d          — sklearn 'distance': closer neighbours weighted more
  3. 1/d²         — sharper inverse: nearest neighbour dominates strongly

Uses the same 80/20 stratified split and PCA fit as day2_knn.py (random_state=42),
so results are directly comparable.
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from python_files.knn_model import load_data, fit_pca, LABEL_NAMES

# ---------------------------------------------------------------------------
# Load data + split (identical to day2_knn.py)
# ---------------------------------------------------------------------------
print("Loading data...")
X, y, X_pred, df, df_pred = load_data()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Fit PCA on training split only — no leakage
pca, X_train_r = fit_pca(X_train, n_components=50)
X_test_r = pca.transform(X_test)

print(f"  Train: {X_train_r.shape}  |  Test: {X_test_r.shape}\n")

# ---------------------------------------------------------------------------
# Define the three weighting schemes
# ---------------------------------------------------------------------------
SCHEMES = [
    ("Uniform  (all = 1)",    "uniform"),
    ("1/d      (current)",    "distance"),
    ("1/d²     (sharper)",    lambda d: 1.0 / (d ** 2 + 1e-9)),
]

class_names = [LABEL_NAMES[i] for i in range(10)]

# ---------------------------------------------------------------------------
# Run comparison
# ---------------------------------------------------------------------------
results = []

for name, weights in SCHEMES:
    knn = KNeighborsClassifier(
        n_neighbors=5,
        metric="cosine",
        algorithm="brute",
        weights=weights,
    )
    knn.fit(X_train_r, y_train)
    y_pred = knn.predict(X_test_r)

    acc = accuracy_score(y_test, y_pred)
    results.append((name, acc, y_pred))

    print("=" * 60)
    print(f"  {name}")
    print(f"  Accuracy: {acc:.4f}")
    print("=" * 60)
    print(classification_report(
        y_test, y_pred,
        labels=list(range(10)),
        target_names=class_names,
        digits=3,
    ))

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  {'Scheme':<28}  {'Accuracy':>8}")
print(f"  {'-'*28}  {'-'*8}")
for name, acc, _ in results:
    marker = "  ← best" if acc == max(r[1] for r in results) else ""
    print(f"  {name:<28}  {acc:.4f}{marker}")
print()
