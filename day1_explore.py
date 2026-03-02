"""
Day 1: Data Exploration & Visualisation
MSIN0025 Scenario Week — Fashion Product Images

Loads product_images.csv, runs sanity checks, reshapes a single row of
784 pixel values into a 28x28 matrix, and displays it with imshow.
Also produces:
  - A representative gallery (3 examples per class)
  - Summary statistics tables for the technical report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Label mapping (per CLAUDE.md Section 12)
# ---------------------------------------------------------------------------
LABEL_NAMES = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}

PIXEL_COLS = [f"pixel_{i}" for i in range(784)]

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
df = pd.read_csv("product_images.csv")
df_pred = pd.read_csv("product_images_for_prediction.csv")

# ---------------------------------------------------------------------------
# 2. Sanity checks & summary tables (for report Section III)
# ---------------------------------------------------------------------------
print("=" * 60)
print("TABLE 1: Dataset Structure")
print("=" * 60)
structure = pd.DataFrame({
    "Dataset": ["product_images.csv", "product_images_for_prediction.csv"],
    "Rows": [len(df), len(df_pred)],
    "Columns": [df.shape[1], df_pred.shape[1]],
    "Has Label": ["Yes", "No"],
    "Missing Values": [df.isnull().sum().sum(), df_pred.isnull().sum().sum()],
})
print(structure.to_string(index=False))

print("\n" + "=" * 60)
print("TABLE 2: Pixel Intensity Summary (product_images.csv)")
print("=" * 60)
pixel_vals = df[PIXEL_COLS].values.flatten()
intensity_summary = pd.DataFrame({
    "Statistic": ["Min", "Max", "Mean", "Std Dev", "Median"],
    "Value": [
        pixel_vals.min(),
        pixel_vals.max(),
        round(pixel_vals.mean(), 2),
        round(pixel_vals.std(), 2),
        round(np.median(pixel_vals), 2),
    ],
})
print(intensity_summary.to_string(index=False))

print("\n" + "=" * 60)
print("TABLE 3: Class Distribution (product_images.csv)")
print("=" * 60)
class_counts = df["label"].value_counts().sort_index()
class_table = pd.DataFrame({
    "Label ID": class_counts.index,
    "Class Name": [LABEL_NAMES[i] for i in class_counts.index],
    "Count": class_counts.values,
    "% of Total": (class_counts.values / len(df) * 100).round(1),
})
print(class_table.to_string(index=False))

print("\n" + "=" * 60)
print("TABLE 4: Per-Class Pixel Intensity Statistics")
print("=" * 60)
rows = []
for label_id in range(10):
    pixels = df[df["label"] == label_id][PIXEL_COLS].values.flatten()
    rows.append({
        "Label": label_id,
        "Class": LABEL_NAMES[label_id],
        "Mean": round(pixels.mean(), 1),
        "Std": round(pixels.std(), 1),
        "Min": pixels.min(),
        "Max": pixels.max(),
    })
per_class = pd.DataFrame(rows)
print(per_class.to_string(index=False))

# ---------------------------------------------------------------------------
# 3. Reshape a single row and display it
# ---------------------------------------------------------------------------
row_index = 0
row = df.iloc[row_index]
label_val = int(row["label"])
label_name = LABEL_NAMES[label_val]

img = row[PIXEL_COLS].values.reshape(28, 28)

plt.figure(figsize=(3, 3))
plt.imshow(img, cmap="gray_r", vmin=0, vmax=255)
plt.title(f"Row {row_index}  |  Label {label_val}: {label_name}")
plt.axis("off")
plt.tight_layout()
plt.savefig("figure1_single_image.png", dpi=150, bbox_inches="tight")
plt.show()

# ---------------------------------------------------------------------------
# 4. Representative gallery: 3 examples per class (10 rows x 3 cols)
# ---------------------------------------------------------------------------
N_EXAMPLES = 3
fig, axes = plt.subplots(10, N_EXAMPLES, figsize=(N_EXAMPLES * 2, 10 * 2))
fig.suptitle("Representative Gallery — 3 Examples per Class", fontsize=14, y=1.01)

for label_id in range(10):
    subset = df[df["label"] == label_id]
    # Space samples evenly across the class so we avoid showing near-duplicates
    indices = np.linspace(0, len(subset) - 1, N_EXAMPLES, dtype=int)
    for col, idx in enumerate(indices):
        sample_img = subset.iloc[idx][PIXEL_COLS].values.reshape(28, 28)
        ax = axes[label_id, col]
        ax.imshow(sample_img, cmap="gray_r", vmin=0, vmax=255)
        ax.axis("off")
        if col == 0:
            ax.set_ylabel(f"{label_id}: {LABEL_NAMES[label_id]}", fontsize=8,
                          rotation=0, labelpad=70, va="center")

plt.tight_layout()
plt.savefig("figure2_gallery.png", dpi=150, bbox_inches="tight")
plt.show()

# ---------------------------------------------------------------------------
# 5. Pixel intensity distribution per class (histogram grid)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 5, figsize=(15, 6), sharey=True)
fig.suptitle("Pixel Intensity Distribution per Class", fontsize=13)

for label_id, ax in zip(range(10), axes.flatten()):
    pixels = df[df["label"] == label_id][PIXEL_COLS].values.flatten()
    ax.hist(pixels, bins=32, range=(0, 255), color="steelblue", edgecolor="none", alpha=0.85)
    ax.set_title(f"{label_id}: {LABEL_NAMES[label_id]}", fontsize=8)
    ax.set_xlabel("Pixel intensity", fontsize=7)
    ax.set_xlim(0, 255)
    ax.tick_params(labelsize=6)

axes[0, 0].set_ylabel("Frequency", fontsize=8)
axes[1, 0].set_ylabel("Frequency", fontsize=8)
plt.tight_layout()
plt.savefig("figure3_intensity_distributions.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nAll figures saved: figure1_single_image.png, figure2_gallery.png, figure3_intensity_distributions.png")
