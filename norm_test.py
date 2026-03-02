"""
norm_test.py
MSIN0025 Scenario Week — L2 Normalisation Test

Compares accuracy with and without L2 normalisation of pixel vectors
before PCA reduction. Config: PCA=125, cosine, weights='distance', k=5.

L2 normalisation divides each image's pixel vector by its Euclidean norm,
so all images have unit length. This removes brightness/exposure differences —
a dark shirt and a light shirt become identical in direction, differing only
in magnitude which cosine already ignores. Effect: cosine distance in PCA
space becomes purely about pixel pattern shape, not overall brightness.

Same 80/20 stratified split as day2_knn.py (random_state=42).
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score, classification_report

from python_files.knn_model import load_data, LABEL_NAMES

print("Loading data...")
X, y, X_pred, df, df_pred = load_data()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

class_names = [LABEL_NAMES[i] for i in range(10)]

CONFIGS = [
    ("No normalisation  (current)", False),
    ("L2 normalisation  (new)",     True),
]

results = []

for name, do_norm in CONFIGS:
    X_tr = normalize(X_train, norm="l2") if do_norm else X_train.copy()
    X_te = normalize(X_test,  norm="l2") if do_norm else X_test.copy()

    pca = PCA(n_components=125, random_state=42)
    X_tr_r = pca.fit_transform(X_tr)
    X_te_r = pca.transform(X_te)

    knn = KNeighborsClassifier(
        n_neighbors=5,
        metric="cosine",
        algorithm="brute",
        weights="distance",
    )
    knn.fit(X_tr_r, y_tr := y_train)
    y_pred = knn.predict(X_te_r)
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

print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  {'Config':<35}  {'Accuracy':>8}")
print(f"  {'-'*35}  {'-'*8}")
for name, acc, _ in results:
    marker = "  ← better" if acc == max(r[1] for r in results) else ""
    print(f"  {name:<35}  {acc:.4f}{marker}")
