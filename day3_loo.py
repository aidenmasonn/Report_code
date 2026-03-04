"""
Day 3: Leave-One-Out (LOO) Cross-Validation
MSIN0025 Scenario Week — Fashion Product Images

LOO is a special case of k-fold cross-validation where k equals the number of
samples. In each fold, one sample is held out as the test set and the model is
trained on the remaining n-1 samples. This provides an essentially unbiased
estimate of generalisation error, but is computationally expensive.

Because a full LOO over all 20,000 labeled images would take many hours, we
use a *stratified random subsample* of 500 images (50 per class). This is
representative of the full dataset's class balance while being feasible to run.

Key design decision — data leakage prevention:
  PCA must be re-fitted inside each LOO fold on the training portion only.
  We achieve this cleanly using a sklearn Pipeline([pca, knn]), which ensures
  the transformer is re-fitted on training data at every fold automatically.

Outputs:
  figure9_loo_per_class.png  — per-class LOO accuracy bar chart
  loo_results.csv            — per-sample predicted labels and correctness flags
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay,
)

from python_files.knn_model import LABEL_NAMES, load_data

# ---------------------------------------------------------------------------
# 0. Load data and draw a stratified subsample
# ---------------------------------------------------------------------------
print("Loading data...")
X, y, X_pred, df, df_pred = load_data()
print(f"  Full labeled set: {X.shape}\n")

# Stratified subsample: 50 images per class (500 total).
# Fixing the random seed makes the experiment fully reproducible.
SAMPLES_PER_CLASS = 50
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

sample_indices = []
for label_id in range(10):
    class_mask = np.where(y == label_id)[0]
    chosen = rng.choice(class_mask, size=SAMPLES_PER_CLASS, replace=False)
    sample_indices.extend(chosen.tolist())

sample_indices = np.array(sample_indices)
X_sample = X[sample_indices]
y_sample = y[sample_indices]

print(f"LOO subsample: {X_sample.shape}  ({SAMPLES_PER_CLASS} images × 10 classes)")
print(f"  Class distribution: {dict(zip(*np.unique(y_sample, return_counts=True)))}\n")

# ---------------------------------------------------------------------------
# 1. Build the Pipeline — PCA(50) → k-NN(k=5, Euclidean, kd_tree)
# ---------------------------------------------------------------------------
# The Pipeline re-fits PCA at each LOO fold on the n-1 training samples,
# then transforms the held-out test sample using those fitted components.
# This is the correct way to prevent data leakage through PCA.

pipe = Pipeline([
    ("pca", PCA(n_components=300, random_state=RANDOM_SEED)),
    ("knn", KNeighborsClassifier(n_neighbors=7, metric="cosine", algorithm="brute", weights="distance")),
])

# ---------------------------------------------------------------------------
# 2. Run LOO cross-validation
# ---------------------------------------------------------------------------
print("=" * 60)
print("LOO Cross-Validation  (n=500, PCA=300, k=7, Cosine/brute, HOG+pixels)")
print("=" * 60)
print("Running 500 LOO folds — this may take 2–5 minutes...\n")

loo = LeaveOneOut()

# cross_val_predict returns the predicted label for each sample when it was
# held out. This is equivalent to running the full LOO loop manually.
y_loo_pred = cross_val_predict(pipe, X_sample, y_sample, cv=loo)

# ---------------------------------------------------------------------------
# 3. Aggregate metrics
# ---------------------------------------------------------------------------
loo_accuracy  = accuracy_score(y_sample, y_loo_pred)
loo_f1_macro  = f1_score(y_sample, y_loo_pred, average="macro")
loo_f1_weight = f1_score(y_sample, y_loo_pred, average="weighted")

print(f"LOO Accuracy        : {loo_accuracy:.4f}")
print(f"LOO Macro F1        : {loo_f1_macro:.4f}")
print(f"LOO Weighted F1     : {loo_f1_weight:.4f}")
print()

label_order = list(range(10))
class_names = [LABEL_NAMES[i] for i in label_order]

print("Per-class classification report:")
print(classification_report(y_sample, y_loo_pred,
                             labels=label_order, target_names=class_names))

# ---------------------------------------------------------------------------
# 4. Per-class accuracy bar chart
# ---------------------------------------------------------------------------
# Compute per-class accuracy manually: correct / total for each class.
per_class_acc = {}
for label_id in label_order:
    mask = y_sample == label_id
    per_class_acc[label_id] = accuracy_score(y_sample[mask], y_loo_pred[mask])

class_labels = [LABEL_NAMES[i] for i in label_order]
class_accs   = [per_class_acc[i] for i in label_order]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(class_labels, class_accs, color="steelblue", edgecolor="white")
ax.axhline(loo_accuracy, color="crimson", linestyle="--",
           label=f"Overall LOO accuracy = {loo_accuracy:.3f}")
ax.set_ylim(0, 1.05)
ax.set_ylabel("LOO Accuracy")
ax.set_title(f"Per-Class LOO Accuracy  (n={len(y_sample)}, k=7, PCA=300, Cosine, HOG+pixels)")
ax.set_xticklabels(class_labels, rotation=30, ha="right")
ax.legend()
ax.grid(axis="y", alpha=0.3)

# Add value labels above each bar
for bar, acc in zip(bars, class_accs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{acc:.2f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig("figure9_loo_per_class.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved figure9_loo_per_class.png\n")

# ---------------------------------------------------------------------------
# 5. LOO Confusion Matrix
# ---------------------------------------------------------------------------
cm = confusion_matrix(y_sample, y_loo_pred, labels=label_order)
fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax, colorbar=True, cmap="Blues", xticks_rotation=45)
ax.set_title(f"LOO Confusion Matrix  (n={len(y_sample)}, k=7, PCA=300, Cosine, HOG+pixels)")
plt.tight_layout()
plt.savefig("figure10_loo_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved figure10_loo_confusion_matrix.png\n")

# ---------------------------------------------------------------------------
# 6. Save per-sample LOO results to CSV
# ---------------------------------------------------------------------------
loo_df = pd.DataFrame({
    "original_index": sample_indices,
    "true_label":     y_sample,
    "true_class":     [LABEL_NAMES[l] for l in y_sample],
    "predicted_label": y_loo_pred,
    "predicted_class": [LABEL_NAMES[l] for l in y_loo_pred],
    "correct":         (y_sample == y_loo_pred).astype(int),
})
loo_df.to_csv("loo_results.csv", index=False)
print(f"Saved loo_results.csv  ({len(loo_df)} rows)\n")

print("=== Day 3 LOO complete ===")
print("Outputs saved:")
print("  figure9_loo_per_class.png")
print("  figure10_loo_confusion_matrix.png")
print("  loo_results.csv")
