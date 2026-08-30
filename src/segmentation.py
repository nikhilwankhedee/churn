"""
Customer segmentation via K-Means with robust fallback and PCA visualisation.
"""
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from src.config import N_CLUSTERS, RANDOM_SEED
from src.run_context import models_dir
from src.utils import get_logger, ensure_dir

logger = get_logger(__name__)


def segment_customers(features: pd.DataFrame) -> pd.DataFrame:
    out_dir = models_dir('segmentation')
    X = features.select_dtypes(include=[np.number]).copy()
    n = len(X)

    if n < 2:
        logger.warning("Insufficient samples for segmentation (%d)", n)
        result = features.copy()
        result['segment'] = 0
        result['pca_x'] = 0.0
        result['pca_y'] = 0.0
        return result

    n_clusters = min(N_CLUSTERS, n - 1) if n > 1 else 1
    if n_clusters < 1:
        n_clusters = 1

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED,
                    n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    if n_clusters > 1:
        try:
            sil = silhouette_score(X_scaled, clusters)
            logger.info("Silhouette score: %.4f", sil)
        except Exception as exc:
            logger.debug("Silhouette score failed: %s", exc)

    with open(os.path.join(out_dir, 'kmeans.pkl'), 'wb') as f:
        pickle.dump(kmeans, f)
    with open(os.path.join(out_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    coords = pca.fit_transform(X_scaled)

    result = features.copy()
    result['segment'] = clusters.astype(int)
    result['pca_x'] = coords[:, 0]
    result['pca_y'] = coords[:, 1]

    with open(os.path.join(out_dir, 'pca.pkl'), 'wb') as f:
        pickle.dump(pca, f)

    logger.info("Segmentation complete — %d clusters", n_clusters)
    return result
