"""
knn_model.py
MSIN0025 Scenario Week — Reusable k-NN + PCA Module

This module contains all core logic for:
  - Loading and preprocessing both datasets
  - Extracting multi-scale HOG + L2-pixel features
  - Fitting a PCA dimensionality reduction pipeline
  - Fitting a k-Nearest Neighbours classifier
  - Querying nearest neighbours for a given image

Designed to be imported by both day2_knn.py (experiments) and the
Dash web application with no modification needed.

Final configuration (empirically optimised):
  Feature : HOG cell=4 (1296 dims) + HOG cell=7 (324 dims) + L2 pixels (784 dims)
            → concatenated and L2-renormalised → 2404 dims
  PCA     : 300 components (~87.9% variance retained)
  Metric  : Cosine
  Weights : Distance (1/d)
  k       : 7

Accuracy history (professor-validated):
  Baseline PCA=50, Euclidean, uniform, k=5            : 84.20%
  + distance weights                                   : 84.35%
  + cosine metric                                      : 84.55%
  + PCA=125                                            : 85.30%
  + L2 normalisation                                   : 85.34%
  + HOG cell=4 + pixels, PCA=250, k=9                 : ~88.50% (test split)
  + HOG cell=4+7 + pixels, PCA=300, k=7 (current)     : ~88.65% (test split)
"""

import numpy as np
import pandas as pd
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize

# ---------------------------------------------------------------------------
# Constants
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
# Feature extraction: multi-scale HOG + L2 pixels
# ---------------------------------------------------------------------------
def extract_features(X_pixels):
    """
    Extract the combined multi-scale HOG + L2-pixel feature vector for each image.

    Pipeline per image:
      1. Compute HOG with 4x4 pixel cells (fine-grained local edges -- seams,
         collar boundaries, strap widths). Produces 1,296 values.
      2. Compute HOG with 7x7 pixel cells (coarse shape -- overall silhouette,
         garment outline). Produces 324 values.
      3. L2-normalise raw pixels (removes brightness bias). Produces 784 values.
      4. Concatenate all three and L2-renormalise the combined vector (2,404 dims).

    Why multi-scale HOG outperforms raw pixels alone:
      - Raw pixels encode brightness patterns but not structural geometry.
      - HOG cell=4 captures fine edge detail (e.g. a shirt collar vs T-shirt neckline).
      - HOG cell=7 captures coarse shape (e.g. a dress silhouette vs a coat silhouette).
      - Together these give the k-NN classifier structural cues that resolve the
        Shirt/T-shirt/Coat/Pullover confusion cluster that limits pixel-only models.

    Parameters
    ----------
    X_pixels : ndarray (n_samples, 784) -- raw pixel values in [0, 255]

    Returns
    -------
    X_feat : ndarray (n_samples, 2404) -- L2-normalised combined feature matrix
    """
    hog4_list = []
    hog7_list = []

    for row in X_pixels:
        img = row.reshape(28, 28)

        # Fine-grained HOG: 7x7 grid of 4x4 cells, each normalised over 2x2 blocks
        h4 = hog(img, pixels_per_cell=(4, 4), cells_per_block=(2, 2),
                 orientations=9, feature_vector=True, channel_axis=None)

        # Coarse HOG: 4x4 grid of 7x7 cells, each normalised over 2x2 blocks
        h7 = hog(img, pixels_per_cell=(7, 7), cells_per_block=(2, 2),
                 orientations=9, feature_vector=True, channel_axis=None)

        hog4_list.append(h4)
        hog7_list.append(h7)

    X_hog4 = normalize(np.array(hog4_list, dtype=np.float32), norm="l2")  # (n, 1296)
    X_hog7 = normalize(np.array(hog7_list, dtype=np.float32), norm="l2")  # (n, 324)
    X_l2   = normalize(X_pixels.astype(np.float32),            norm="l2")  # (n, 784)

    # Concatenate and re-normalise so no single component dominates cosine distance
    X_feat = normalize(np.hstack([X_hog4, X_hog7, X_l2]), norm="l2")      # (n, 2404)
    return X_feat


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(labeled_path="product_images.csv",
              unlabeled_path="product_images_for_prediction.csv"):
    """
    Load both datasets from CSV and extract multi-scale HOG + pixel features.

    Returns
    -------
    X      : ndarray (20000, 2404) -- feature vectors for labeled training data
    y      : ndarray (20000,)      -- integer class labels 0-9
    X_pred : ndarray (10000, 2404) -- feature vectors for unlabeled prediction data
    df     : DataFrame             -- full labeled DataFrame (includes 'label' col)
    df_pred: DataFrame             -- full unlabeled DataFrame
    """
    df      = pd.read_csv(labeled_path)
    df_pred = pd.read_csv(unlabeled_path)

    X_raw      = df[PIXEL_COLS].values.astype(np.float32)
    y          = df["label"].values.astype(int)
    X_pred_raw = df_pred[PIXEL_COLS].values.astype(np.float32)

    print("  Extracting features for labeled data ...", flush=True)
    X      = extract_features(X_raw)

    print("  Extracting features for unlabeled data ...", flush=True)
    X_pred = extract_features(X_pred_raw)

    return X, y, X_pred, df, df_pred


# ---------------------------------------------------------------------------
# PCA pipeline
# ---------------------------------------------------------------------------
def fit_pca(X, n_components=300):
    """
    Fit a PCA on training data and return the reduced representation.

    Parameters
    ----------
    X           : ndarray (n_samples, 2404) -- multi-scale HOG + pixel features
    n_components: int -- number of principal components to retain.
                  Default 300 retains ~87.9% of variance in the HOG+pixel space.
                  Empirically optimal: more components add noise, fewer lose signal.

    Returns
    -------
    pca       : fitted sklearn PCA object
    X_reduced : ndarray (n_samples, n_components)
    """
    pca = PCA(n_components=n_components, random_state=42)
    X_reduced = pca.fit_transform(X)
    return pca, X_reduced


# ---------------------------------------------------------------------------
# k-NN classifier
# ---------------------------------------------------------------------------
def fit_knn(X_reduced, y, n_neighbors=7, metric="cosine", algorithm="auto",
            weights="distance"):
    """
    Fit a k-Nearest Neighbours classifier on PCA-reduced data.

    Parameters
    ----------
    X_reduced  : ndarray (n_samples, n_components)
    y          : ndarray (n_samples,) -- integer labels
    n_neighbors: int -- number of neighbours. Default 7, empirically optimal for
                 the HOG+pixel feature space (k=7 peaks on both 80/20 and 90/10
                 splits; larger k dilutes the vote with distant, less reliable
                 neighbours).
    metric     : str -- 'cosine' (default). Cosine distance is scale-invariant
                 and outperforms Euclidean/Manhattan on L2-normalised features.
    algorithm  : str -- 'auto', 'brute', 'kd_tree', or 'ball_tree'.
                 Cosine distance requires brute-force; enforced automatically.
    weights    : str -- 'distance' (default). Weights neighbours by 1/distance
                 so closer neighbours have proportionally more influence.

    Returns
    -------
    knn : fitted KNeighborsClassifier
    """
    if metric == "cosine" and algorithm != "brute":
        algorithm = "brute"

    knn = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        metric=metric,
        algorithm=algorithm,
        weights=weights,
    )
    knn.fit(X_reduced, y)
    return knn


# ---------------------------------------------------------------------------
# Query: find nearest neighbours for a single image
# ---------------------------------------------------------------------------
def find_neighbours(knn, pca, query_vector, k=7):
    """
    Find the k nearest neighbours for a single raw 784-pixel query image.

    Parameters
    ----------
    knn          : fitted KNeighborsClassifier
    pca          : fitted PCA object
    query_vector : ndarray (784,) -- raw pixel values for a single image
    k            : int -- number of neighbours to return (default 7)

    Returns
    -------
    indices         : ndarray (k,) -- row indices in the training set
    distances       : ndarray (k,) -- distances to each neighbour
    predicted_label : int          -- predicted class
    """
    # Extract features for the single query image then project into PCA space
    query_feat    = extract_features(query_vector.reshape(1, -1))
    query_reduced = pca.transform(query_feat)
    distances, indices = knn.kneighbors(query_reduced, n_neighbors=k)
    predicted_label = int(knn.predict(query_reduced)[0])
    return indices[0], distances[0], predicted_label


# ---------------------------------------------------------------------------
# Prediction confidence helper
# ---------------------------------------------------------------------------
def prediction_confidence(knn, query_reduced):
    """
    Return the highest class probability for a PCA-projected query.

    Parameters
    ----------
    knn           : fitted KNeighborsClassifier
    query_reduced : ndarray (1, n_components) -- already PCA-projected query

    Returns
    -------
    label      : int   -- predicted class
    confidence : float -- proportion of distance-weighted votes for that class
    """
    proba = knn.predict_proba(query_reduced)[0]
    label = int(np.argmax(proba))
    confidence = float(proba[label])
    return label, confidence
