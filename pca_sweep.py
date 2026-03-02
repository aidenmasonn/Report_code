"""
pca_sweep.py
MSIN0025 Scenario Week — PCA Component Sweep

Tests accuracy at different PCA component counts (cosine, k=5, weights='distance')
to find the optimal n_components before committing to a change in knn_model.py.

Same 80/20 stratified split as day2_knn.py (random_state=42).
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from python_files.knn_model import load_data, LABEL_NAMES

print("Loading data...")
X, y, X_pred, df, df_pred = load_data()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

N_COMPONENTS_LIST = [50, 75, 100, 125, 150]
class_names = [LABEL_NAMES[i] for i in range(10)]

results = []

for n in N_COMPONENTS_LIST:
    pca = PCA(n_components=n, random_state=42)
    X_train_r = pca.fit_transform(X_train)
    X_test_r = pca.transform(X_test)

    var = np.sum(pca.explained_variance_ratio_)

    knn = KNeighborsClassifier(
        n_neighbors=5,
        metric="cosine",
        algorithm="brute",
        weights="distance",
    )
    knn.fit(X_train_r, y_train)
    y_pred = knn.predict(X_test_r)
    acc = accuracy_score(y_test, y_pred)

    results.append((n, var, acc, y_pred))
    print(f"  n_components={n:>3}  |  variance={var:.1%}  |  accuracy={acc:.4f}")

# Full per-class report for best configuration
best = max(results, key=lambda r: r[2])
best_n, best_var, best_acc, best_pred = best

print()
print("=" * 60)
print(f"BEST: n_components={best_n}  (variance={best_var:.1%},  accuracy={best_acc:.4f})")
print("=" * 60)
print(classification_report(
    y_test, best_pred,
    labels=list(range(10)),
    target_names=class_names,
    digits=3,
))

print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  {'n_components':>12}  {'Variance':>9}  {'Accuracy':>9}")
print(f"  {'-'*12}  {'-'*9}  {'-'*9}")
for n, var, acc, _ in results:
    marker = "  ← best" if acc == best_acc else ""
    print(f"  {n:>12}  {var:>8.1%}  {acc:>9.4f}{marker}")
