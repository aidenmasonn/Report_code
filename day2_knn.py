"""
Day 2: k-NN Implementation, PCA, and Optimisation
MSIN0025 Scenario Week — Fashion Product Images

Experiments run in this script:
  1. PCA explained variance curve — choose n_components
  2. Distance metric comparison — Euclidean vs Manhattan vs Cosine (accuracy + speed)
  3. Algorithm timing benchmark — brute vs kd_tree vs ball_tree
  4. Nearest neighbours visualisation — query image + 5 neighbours per class
  5. Predict labels for the 10,000 unlabeled images and save to predictions.csv
  6. k-value sweep — accuracy vs k to justify choice of k=5
  7. Confusion matrix + F1/precision/recall — full evaluation metrics
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
    classification_report, f1_score,
)

from python_files.knn_model import (
    LABEL_NAMES, PIXEL_COLS,
    load_data, fit_pca, fit_knn, find_neighbours, prediction_confidence,
    extract_features,
)

# ---------------------------------------------------------------------------
# 0. Load data
# ---------------------------------------------------------------------------
print("Loading data...")
X, y, X_pred, df, df_pred = load_data()
print(f"  Training set : {X.shape}  |  Labels: {y.shape}")
print(f"  Prediction set: {X_pred.shape}\n")

# Train / test split (80/20) — used for metric and algorithm comparisons.
# PCA is fitted on X_train only to avoid data leakage into the test set.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------------------------
# 1. PCA Explained Variance Curve
# ---------------------------------------------------------------------------
print("=" * 60)
print("SECTION 1: PCA Explained Variance")
print("=" * 60)

# Fit PCA with 350 components on training data to inspect the elbow.
# Feature space is now 2,404-dim (HOG cell=4 + HOG cell=7 + L2 pixels).
# Empirical optimum is 300 components (~87.9% variance retained).
pca_explore = PCA(n_components=350, random_state=42)
pca_explore.fit(X_train)

cumvar = np.cumsum(pca_explore.explained_variance_ratio_)
n_components_fitted = len(cumvar)
n_95 = int(np.searchsorted(cumvar, 0.95)) + 1   # components needed for 95%
n_chosen = 300                                   # empirically optimal value

print(f"  Components for 95% variance : {n_95}")
print(f"  Variance retained at 300 PCs: {cumvar[299]:.1%}")
# Guard: only print the n_95 row if it falls within the fitted range
if n_95 <= n_components_fitted:
    print(f"  Variance retained at {n_95} PCs: {cumvar[n_95-1]:.1%}\n")
else:
    print(f"  (95% threshold beyond {n_components_fitted} fitted components)\n")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1, n_components_fitted + 1), cumvar, color="steelblue", lw=2)
ax.axhline(0.95, color="crimson", linestyle="--", label="95% threshold")
ax.axvline(n_95, color="crimson", linestyle=":", alpha=0.7)
ax.axvline(n_chosen, color="darkorange", linestyle="--",
           label=f"Chosen: {n_chosen} PCs ({cumvar[n_chosen-1]:.1%})")
ax.set_xlabel("Number of principal components")
ax.set_ylabel("Cumulative explained variance")
ax.set_title("PCA: Cumulative Explained Variance (HOG cell=4+7 + L2 pixels)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figure4_pca_variance.png", dpi=150, bbox_inches="tight")
plt.show()

# ---------------------------------------------------------------------------
# 2. Distance Metric Comparison
# ---------------------------------------------------------------------------
print("=" * 60)
print("SECTION 2: Distance Metric Comparison  (k=7, PCA=300, test n=4000)")
print("=" * 60)

# Fit PCA(300) on training split — reused across all metric experiments
pca50, X_train_r = fit_pca(X_train, n_components=300)
X_test_r = pca50.transform(X_test)

METRICS = [
    ("Euclidean", "euclidean", "kd_tree"),
    ("Manhattan", "manhattan", "kd_tree"),
    ("Cosine",    "cosine",    "brute"),   # cosine requires brute-force
]

metric_results = []
for name, metric, algo in METRICS:
    t0 = time.perf_counter()
    knn = fit_knn(X_train_r, y_train, n_neighbors=7, metric=metric, algorithm=algo)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = knn.predict(X_test_r)
    query_time = time.perf_counter() - t0

    acc = accuracy_score(y_test, y_pred)
    metric_results.append({
        "Metric": name,
        "Algorithm": algo,
        "Accuracy": f"{acc:.4f}",
        "Fit time (s)": f"{fit_time:.3f}",
        "Query time (s)": f"{query_time:.3f}",
    })
    print(f"  {name:<12} | acc={acc:.4f} | fit={fit_time:.3f}s | query={query_time:.3f}s")

print()
metric_table = pd.DataFrame(metric_results)
print(metric_table.to_string(index=False))
print()

# Identify best metric by accuracy for use in remaining sections
best_metric_row = max(metric_results, key=lambda r: float(r["Accuracy"]))
best_metric = best_metric_row["Metric"].lower()
best_algo_for_metric = best_metric_row["Algorithm"]
print(f"  Best metric: {best_metric_row['Metric']} "
      f"(accuracy={best_metric_row['Accuracy']})\n")

# ---------------------------------------------------------------------------
# 3. Algorithm Timing Benchmark (Euclidean, PCA=50 vs raw 784)
# ---------------------------------------------------------------------------
print("=" * 60)
print("SECTION 3: Algorithm Timing Benchmark  (k=7, Euclidean)")
print("=" * 60)

# Raw 2404-dim brute-force as unoptimised baseline
BENCHMARKS = [
    ("Brute (raw 2404 dims)", "euclidean", "brute",    X_train,   X_test),
    ("Brute (PCA 300 dims)",  "euclidean", "brute",    X_train_r, X_test_r),
    ("KD-tree (PCA 300 dims)","euclidean", "kd_tree",  X_train_r, X_test_r),
    ("Ball-tree (PCA 300 dims)","euclidean","ball_tree",X_train_r, X_test_r),
]

timing_results = []
for label, metric, algo, X_tr, X_te in BENCHMARKS:
    t0 = time.perf_counter()
    knn_b = fit_knn(X_tr, y_train, n_neighbors=7, metric=metric, algorithm=algo)
    fit_t = time.perf_counter() - t0

    t0 = time.perf_counter()
    knn_b.predict(X_te)
    query_t = time.perf_counter() - t0

    timing_results.append({
        "Configuration": label,
        "Fit time (s)": f"{fit_t:.3f}",
        "Query time (s)": f"{query_t:.3f}",
    })
    print(f"  {label:<28} | fit={fit_t:.3f}s | query={query_t:.3f}s")

timing_table = pd.DataFrame(timing_results)
print()
print(timing_table.to_string(index=False))
print()

# Bar chart of query times
fig, ax = plt.subplots(figsize=(8, 4))
configs = [r["Configuration"] for r in timing_results]
qtimes = [float(r["Query time (s)"]) for r in timing_results]
bars = ax.barh(configs, qtimes, color=["#d9534f", "#5bc0de", "#5cb85c", "#f0ad4e"])
ax.set_xlabel("Query time (seconds) — 4,000 test images")
ax.set_title("Algorithm Timing Benchmark (k=5, Euclidean)")
for bar, val in zip(bars, qtimes):
    ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}s", va="center", fontsize=9)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("figure5_timing_benchmark.png", dpi=150, bbox_inches="tight")
plt.show()

# ---------------------------------------------------------------------------
# 4. Nearest Neighbours Visualisation (one query per class)
# ---------------------------------------------------------------------------
print("=" * 60)
print("SECTION 4: Nearest Neighbours Visualisation")
print("=" * 60)

# Use the best metric found in Section 2 for the final fitted k-NN
knn_best = fit_knn(X_train_r, y_train, n_neighbors=7,
                   metric=best_metric, algorithm=best_algo_for_metric)

# Keep raw pixel arrays for display (imshow needs original 0-255 intensities)
X_train_raw = df[PIXEL_COLS].values[
    np.where(np.isin(np.arange(len(y)), train_test_split(
        np.arange(len(y)), test_size=0.2, random_state=42, stratify=y
    )[0]))[0]
]

K_SHOW = 7
fig = plt.figure(figsize=(2.2 * (K_SHOW + 1), 2.2 * 10))
gs = gridspec.GridSpec(10, K_SHOW + 1, figure=fig,
                       hspace=0.5, wspace=0.1)
fig.suptitle(
    f"Query image + {K_SHOW} nearest neighbours  "
    f"(metric={best_metric}, PCA=300, HOG+pixels)",
    fontsize=13, y=1.005,
)

for row, label_id in enumerate(range(10)):
    # Pick one query image from X_test that belongs to this class
    class_indices = np.where(y_test == label_id)[0]
    query_idx = class_indices[0]
    # X_test holds HOG+pixel features — use df to get raw pixels for display
    query_raw_pixels = df[PIXEL_COLS].values[
        train_test_split(np.arange(len(y)), test_size=0.2,
                         random_state=42, stratify=y)[1]
    ][query_idx]
    query_feat = X_test[query_idx]   # HOG+pixel feature vector

    # Find neighbours using pre-extracted features (pca50 is now pca300)
    neighbour_idxs, distances, predicted = find_neighbours(
        knn_best, pca50, query_raw_pixels, k=K_SHOW
    )

    # --- Query image (leftmost column, highlighted) ---
    ax_q = fig.add_subplot(gs[row, 0])
    ax_q.imshow(query_raw_pixels.reshape(28, 28), cmap="gray_r", vmin=0, vmax=255)
    ax_q.set_title(f"QUERY\n{label_id}: {LABEL_NAMES[label_id]}", fontsize=7)
    for spine in ax_q.spines.values():
        spine.set_edgecolor("darkorange")
        spine.set_linewidth(2)
    ax_q.set_xticks([])
    ax_q.set_yticks([])

    # --- Neighbour images (display raw pixels from training set) ---
    train_indices = train_test_split(
        np.arange(len(y)), test_size=0.2, random_state=42, stratify=y
    )[0]
    for col, (nidx, dist) in enumerate(zip(neighbour_idxs, distances), start=1):
        ax_n = fig.add_subplot(gs[row, col])
        neighbour_pixels = df[PIXEL_COLS].values[train_indices[nidx]]
        ax_n.imshow(neighbour_pixels.reshape(28, 28), cmap="gray_r", vmin=0, vmax=255)
        nlabel = y_train[nidx]
        ax_n.set_title(
            f"#{col}  {nlabel}: {LABEL_NAMES[nlabel]}\nd={dist:.3f}",
            fontsize=6,
        )
        ax_n.axis("off")

print("  Nearest neighbours visualisation complete.")

plt.savefig("figure6_nearest_neighbours.png", dpi=150, bbox_inches="tight")
plt.show()

# ---------------------------------------------------------------------------
# 5. Predict labels for unlabeled data
# ---------------------------------------------------------------------------
print("=" * 60)
print("SECTION 5: Predicting Labels for Unlabeled Data")
print("=" * 60)

# Fit final k-NN on ALL labeled data (not just the training split) for best accuracy
print("  Fitting final k-NN on full labeled dataset (n=20,000)...")
pca_final, X_full_r = fit_pca(X, n_components=300)
knn_final = fit_knn(X_full_r, y, n_neighbors=7,
                    metric=best_metric, algorithm=best_algo_for_metric)

# Project unlabeled data into the same PCA space
X_pred_r = pca_final.transform(X_pred)

print("  Running predictions...")
predicted_labels = knn_final.predict(X_pred_r)

# Compute per-row confidence — single batched call (avoids slow Python loop)
print("  Computing confidence scores...")
proba_all = knn_final.predict_proba(X_pred_r)          # (10000, 10)
confidences = proba_all.max(axis=1).tolist()            # highest class probability per row

predictions_df = pd.DataFrame({
    "row_index": range(len(predicted_labels)),
    "predicted_label": predicted_labels,
    "predicted_class": [LABEL_NAMES[l] for l in predicted_labels],
    "confidence": [round(c, 4) for c in confidences],
})

predictions_df.to_csv("predictions.csv", index=False)
print(f"  Saved predictions.csv  ({len(predictions_df):,} rows)")

# Generate submission.csv — single "label" column for professor validation
predictions_df[["predicted_label"]].rename(
    columns={"predicted_label": "label"}
).to_csv("submission.csv", index=False)
print("  Saved submission.csv  (label column only)\n")

print("Prediction summary:")
summary = (
    predictions_df.groupby(["predicted_label", "predicted_class"])
    .size()
    .reset_index(name="count")
)
summary["mean_confidence"] = (
    predictions_df.groupby("predicted_label")["confidence"]
    .mean()
    .round(3)
    .values
)
print(summary.to_string(index=False))

# ---------------------------------------------------------------------------
# 6. k-Value Sweep — accuracy vs k to justify the choice of k=5
# ---------------------------------------------------------------------------
print("=" * 60)
print("SECTION 6: k-Value Sweep  (best metric, PCA=300, test n=4000)")
print("=" * 60)

# Reuse the same 80/20 split and PCA(300) fitted in Section 2 — no leakage.
# Test a range of odd k values to avoid ties in majority-vote classification.
K_VALUES = [1, 3, 5, 7, 9, 11, 13, 15]
k_accuracies = []

for k in K_VALUES:
    knn_k = fit_knn(X_train_r, y_train, n_neighbors=k,
                    metric=best_metric, algorithm=best_algo_for_metric)
    y_pred_k = knn_k.predict(X_test_r)
    acc_k = accuracy_score(y_test, y_pred_k)
    k_accuracies.append(acc_k)
    print(f"  k={k:>2}  |  accuracy={acc_k:.4f}")

print()

# Table
k_sweep_df = pd.DataFrame({"k": K_VALUES, "Accuracy": [f"{a:.4f}" for a in k_accuracies]})
print(k_sweep_df.to_string(index=False))
print()

# Plot accuracy vs k
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(K_VALUES, k_accuracies, marker="o", color="steelblue", lw=2, ms=7)
ax.axvline(7, color="darkorange", linestyle="--", label="Chosen k=7")
ax.set_xlabel("Number of neighbours (k)")
ax.set_ylabel("Test accuracy")
ax.set_title(f"k-Value Sweep  (metric={best_metric}, PCA=300, HOG+pixels)")
ax.set_xticks(K_VALUES)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figure7_k_sweep.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Saved figure7_k_sweep.png\n")

# ---------------------------------------------------------------------------
# 7. Confusion Matrix + F1 / Precision / Recall
# ---------------------------------------------------------------------------
print("=" * 60)
print("SECTION 7: Confusion Matrix + Classification Report  (k=7, best metric)")
print("=" * 60)

# Use k=7 and the best metric — the chosen final classifier configuration.
knn_eval = fit_knn(X_train_r, y_train, n_neighbors=7,
                   metric=best_metric, algorithm=best_algo_for_metric)
y_pred_eval = knn_eval.predict(X_test_r)

# --- Scalar summary metrics ---
acc_eval  = accuracy_score(y_test, y_pred_eval)
f1_macro  = f1_score(y_test, y_pred_eval, average="macro")
f1_weight = f1_score(y_test, y_pred_eval, average="weighted")

print(f"  Accuracy        : {acc_eval:.4f}")
print(f"  Macro F1        : {f1_macro:.4f}")
print(f"  Weighted F1     : {f1_weight:.4f}")
print()

# --- Per-class precision, recall, F1 ---
label_order = list(range(10))
class_names = [LABEL_NAMES[i] for i in label_order]
print("  Classification report (per class):")
print(classification_report(y_test, y_pred_eval,
                             labels=label_order, target_names=class_names))

# --- Confusion matrix heatmap ---
cm = confusion_matrix(y_test, y_pred_eval, labels=label_order)
fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax, colorbar=True, cmap="Blues", xticks_rotation=45)
ax.set_title(f"Confusion Matrix  (k=7, metric={best_metric}, PCA=300, HOG+pixels, n_test=4000)")
plt.tight_layout()
plt.savefig("figure8_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Saved figure8_confusion_matrix.png\n")

print("\n=== Day 2 complete ===")
print("Outputs saved:")
print("  figure4_pca_variance.png")
print("  figure5_timing_benchmark.png")
print("  figure6_nearest_neighbours.png")
print("  figure7_k_sweep.png")
print("  figure8_confusion_matrix.png")
print("  predictions.csv")
