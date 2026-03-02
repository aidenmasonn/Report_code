# Report Code — MSIN0025 Scenario Week

This folder contains all the Python code behind the results, figures, and tables in our report. Each file corresponds to a specific section of the report and can be run independently to reproduce the outputs.

> **Note for report writers:** You do not need to run the code yourself. This document tells you which file produced which figure or table, what the code was trying to find out, and how to describe what it does in plain English.

---

## How the code is structured

All scripts depend on one shared foundation file (`python_files/knn_model.py`) that handles loading the data and converting images into numerical features. The individual scripts then import from it and each focus on one specific experiment or set of results.

---

## File-by-file guide

---

### `python_files/knn_model.py` — The core pipeline (Appendix)

**Report section:** Appendix (reproducibility) — referenced throughout Methods and Implementation.

This is the engine that all other scripts rely on. It handles three things:

1. **Loading the data** — reads the two CSV files (labeled training images and unlabeled prediction images) and stores them in a format the classifier can use.
2. **Feature extraction** — converts each raw 28×28 pixel image into a richer numerical description. Instead of feeding the raw pixel values straight into the classifier, it computes HOG (Histogram of Oriented Gradients) features at two scales — a fine scale (4×4 pixel cells) that picks up small edge details like collar shapes and strap boundaries, and a coarse scale (7×7 pixel cells) that captures the overall silhouette — then combines these with the normalised pixel values. This produces a 2,404-number description per image.
3. **Fitting PCA and k-NN** — provides reusable functions for applying PCA dimensionality reduction and fitting the k-NN classifier so every script uses the exact same procedure.

The final configuration stored here is: **HOG (fine + coarse) + normalised pixels → PCA 300 components → cosine k-NN, k=7, distance weighting.**

---

### `day1_explore.py` — Dataset overview (Section 3: Data)

**Produces:** `figure1_single_image.png`, `figure2_gallery.png`, `figure3_intensity_distributions.png`, and four printed summary tables.

**What it does:** This is the data exploration script. It loads the labeled dataset and answers the basic questions a reader needs before the classifier is introduced: how many images are there, how are they distributed across the 10 clothing categories, and what do they actually look like?

Specifically it produces:
- **Table 1** — the size and structure of both datasets (20,000 labeled, 10,000 unlabeled)
- **Table 2** — pixel intensity statistics (min, max, mean, standard deviation) across the whole dataset
- **Table 3** — how many images belong to each clothing category and what percentage that represents
- **Table 4** — the same intensity statistics broken down per category, showing that some categories (e.g. Bag) are visually much brighter than others (e.g. T-shirt)
- **Figure 1** — a single example image shown as a greyscale picture
- **Figure 2** — a 10×3 gallery showing three representative examples from each category
- **Figure 3** — pixel intensity histograms for each category, illustrating how visually distinct (or not) different garment types are

---

### `day2_knn.py` — Main k-NN experiments (Sections 4, 5 & 6)

**Produces:** `figure4_pca_variance.png`, `figure5_timing_benchmark.png`, `figure6_nearest_neighbours.png`, `figure7_k_sweep.png`, `figure8_confusion_matrix.png`, and several printed tables.

**What it does:** This is the central experiment script that works through the full classifier design process step by step. It uses an 80/20 train/test split throughout (80% of the 20,000 labeled images to train, 20% to evaluate).

- **Section 1 — PCA Explained Variance (Figure 4):** Fits PCA with up to 350 components and plots the cumulative variance explained. The curve flattens around 300 components (retaining ~87.9% of variance), which is why 300 was chosen — adding more components captures noise rather than signal.

- **Section 2 — Distance Metric Comparison:** Tests Euclidean, Manhattan, and Cosine distance. Prints a table of accuracy and speed for each. Cosine distance performs best on these L2-normalised features because it measures the *angle* between feature vectors, ignoring overall magnitude — two identical-shaped garments that happen to be photographed at different brightness levels will still appear similar.

- **Section 3 — Algorithm Timing Benchmark (Figure 5):** Compares four search methods (brute-force on raw features, brute-force on PCA-reduced features, KD-tree, ball-tree) for how long they take to classify 4,000 test images. This justifies the choice of search algorithm in the report's Implementation section.

- **Section 4 — Nearest Neighbours Visualisation (Figure 6):** For one query image from each category, displays the 7 nearest neighbours the classifier retrieved from the training set. This is a visual sanity check — the neighbours should look like the same type of garment, and where they do not, it reveals the classifier's main confusion points (e.g. Shirt vs T-shirt).

- **Section 5 — Predictions:** Runs the final classifier on all 10,000 unlabeled images and saves `predictions.csv`.

- **Section 6 — k-Value Sweep (Figure 7):** Tests k from 1 to 15 and plots accuracy against k. The curve peaks at k=7, justifying that choice.

- **Section 7 — Confusion Matrix and Classification Report (Figure 8):** Shows which categories the classifier confuses most often. The confusion matrix is a 10×10 grid where each cell shows how many test images of one true category were predicted as another. The classification report provides per-category precision, recall, and F1 score.

---

### `pca_sweep.py` — Choosing the number of PCA components (Section 5: Implementation)

**Produces:** printed accuracy table.

**What it does:** Tests five values of PCA components — 50, 75, 100, 125, 150 — keeping everything else fixed (cosine distance, k=5, distance weighting). Reports the accuracy and the proportion of variance retained for each. The results justify why 300 components (tested in `day2_knn.py`) was chosen as the sweet spot: too few components discard useful information, too many retain noise.

This script answers the report requirement to explain *how* the PCA parameter was chosen rather than just asserting it.

---

### `norm_test.py` — Effect of L2 normalisation (Section 5: Implementation)

**Produces:** printed accuracy comparison table.

**What it does:** Runs the classifier twice — once on raw pixel values, once on L2-normalised pixel values — and compares the accuracy. L2 normalisation divides each image's pixel vector by its overall brightness so that a dark photograph and a bright photograph of the same garment are treated as equivalent by the cosine distance function. The result shows whether this pre-processing step actually helps.

---

### `weight_comparison.py` — Choosing how neighbours vote (Section 5: Implementation)

**Produces:** printed accuracy table with per-class breakdown.

**What it does:** Compares three ways of combining the votes of the k nearest neighbours when making a prediction:
- **Uniform** — every neighbour has an equal vote regardless of how far away it is
- **1/d (distance weighting)** — closer neighbours are weighted more heavily in proportion to their distance
- **1/d² (sharper weighting)** — an even more aggressive version where the nearest neighbour dominates strongly

The results justify the choice of distance weighting (`1/d`) in the final classifier.

---

### `hog_sweep.py` — Comparing feature extraction approaches (Section 6: Results)

**Produces:** printed summary table.

**What it does:** Systematically compares different ways of describing the images before feeding them to the classifier. It tests HOG features with small cells (cell=4, fine detail), HOG with larger cells (cell=7, coarser shape), and HOG combined with raw pixels — at various PCA component counts and values of k. Both an 80/20 and a 90/10 train/test split are used to check the results are stable.

This is the evidence that the combination of fine HOG + coarse HOG + normalised pixels outperforms either HOG or pixels alone.

---

### `hog_k_sweep.py` — Fine-tuning k on the best feature set (Section 6: Results)

**Produces:** printed summary table.

**What it does:** Takes the best feature configuration found in `hog_sweep.py` (HOG cell=4 + L2 pixels, PCA=150) and runs a focused sweep over k ∈ {3, 5, 7, 9, 11} on both 80/20 and 90/10 splits. This confirms which k value is optimal specifically for this feature representation before finalising the pipeline.

---

### `day3_loo.py` — Leave-One-Out cross-validation (Section 6: Results)

**Produces:** `figure9_loo_per_class.png`, `figure10_loo_confusion_matrix.png`, `loo_results.csv`.

**What it does:** Runs Leave-One-Out (LOO) cross-validation — the most rigorous way to evaluate a classifier on limited data.

In a standard evaluation, you split the data once into training and test sets. LOO instead repeats this process 500 times (on a representative subsample of 500 images, 50 per category): each time, one image is held back as the test case and the classifier is trained on the remaining 499. The accuracy reported is the average across all 500 trials.

This is important because it gives an essentially unbiased estimate of how the classifier will perform on genuinely new data. A key technical detail — PCA is re-fitted from scratch inside each fold, not once in advance, so the test image never influences the dimensionality reduction. Without this precaution, the results would be artificially inflated.

The per-class bar chart (Figure 9) shows which clothing categories the classifier finds hardest (typically Shirt and Pullover), and the LOO confusion matrix (Figure 10) mirrors the standard confusion matrix but under this stricter evaluation.

---

## Figures and tables at a glance

| Output file | Produced by | Report section |
|---|---|---|
| `figure1_single_image.png` | `day1_explore.py` | Section 3 (Data) |
| `figure2_gallery.png` | `day1_explore.py` | Section 3 (Data) |
| `figure3_intensity_distributions.png` | `day1_explore.py` | Section 3 (Data) |
| Tables 1–4 (dataset stats) | `day1_explore.py` | Section 3 (Data) |
| `figure4_pca_variance.png` | `day2_knn.py` | Section 5 (Implementation) |
| `figure5_timing_benchmark.png` | `day2_knn.py` | Section 5 (Implementation) |
| `figure6_nearest_neighbours.png` | `day2_knn.py` | Section 5 (Implementation) |
| `figure7_k_sweep.png` | `day2_knn.py` | Section 6 (Results) |
| `figure8_confusion_matrix.png` | `day2_knn.py` | Section 6 (Results) |
| Metric comparison table | `day2_knn.py` | Section 6 (Results) |
| PCA component accuracy table | `pca_sweep.py` | Section 5 (Implementation) |
| Normalisation comparison table | `norm_test.py` | Section 5 (Implementation) |
| Weighting scheme table | `weight_comparison.py` | Section 5 (Implementation) |
| HOG vs pixels comparison table | `hog_sweep.py` | Section 6 (Results) |
| k sweep on HOG features table | `hog_k_sweep.py` | Section 6 (Results) |
| `figure9_loo_per_class.png` | `day3_loo.py` | Section 6 (Results) |
| `figure10_loo_confusion_matrix.png` | `day3_loo.py` | Section 6 (Results) |
| LOO accuracy / F1 metrics | `day3_loo.py` | Section 6 (Results) |
