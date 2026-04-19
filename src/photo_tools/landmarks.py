"""
landmarks.py — FAISS-based landmark lookup using CLIP embeddings.

Loads a landmarks.json database (built by build_landmarks.py) and provides
fast cosine-similarity search filtered by GPS radius.
"""

import json
import logging
import math
from pathlib import Path

import numpy as np

from photo_tools.config import get_config

log = logging.getLogger("landmarks")


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in kilometers."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class LandmarkIndex:
    """FAISS-backed landmark lookup using CLIP embeddings."""

    def __init__(self, landmarks_path: Path | None = None):
        cfg = get_config()
        if landmarks_path is None:
            landmarks_path = Path(cfg.landmarks.default_path).expanduser()

        log.info("Loading landmark database from %s ...", landmarks_path)
        with open(landmarks_path) as f:
            data = json.load(f)

        self.model_id = data["model"]
        self.landmarks = data["landmarks"]
        self.names = [lm["name"] for lm in self.landmarks]
        self.lats = np.array([lm["lat"] for lm in self.landmarks], dtype=np.float64)
        self.lons = np.array([lm["lon"] for lm in self.landmarks], dtype=np.float64)

        self._embeddings = np.array(
            [lm["embedding"] for lm in self.landmarks], dtype=np.float32,
        )

        log.info("Loaded %d landmarks (dim=%d)", len(self.landmarks), self._embeddings.shape[1])

    def lookup(
        self,
        embedding: np.ndarray,
        lat: float,
        lon: float,
        radius_km: float | None = None,
        threshold: float | None = None,
    ) -> tuple[str | None, list[tuple[str, float]]]:
        """Find the best matching landmark for an image embedding.

        Returns (best_name_or_None, top_candidates). top_candidates is a
        list of (name, score) sorted by score descending; empty if no
        landmarks are within radius_km. The best match is only returned
        when its score meets `threshold`, but top_candidates is always
        populated when there are nearby landmarks (useful for logging).
        """
        import faiss

        cfg = get_config()
        if radius_km is None:
            radius_km = cfg.landmarks.radius_km
        if threshold is None:
            threshold = cfg.landmarks.threshold

        embedding = embedding.reshape(1, -1).astype(np.float32)

        distances = np.array([
            _haversine_km(lat, lon, self.lats[i], self.lons[i])
            for i in range(len(self.landmarks))
        ])
        nearby_indices = np.where(distances <= radius_km)[0]

        if len(nearby_indices) == 0:
            log.debug("No landmarks within %.0f km of (%.4f, %.4f)",
                      radius_km, lat, lon)
            return None, []

        nearby_embeddings = self._embeddings[nearby_indices]
        dim = nearby_embeddings.shape[1]
        tmp_index = faiss.IndexFlatIP(dim)
        tmp_index.add(nearby_embeddings)

        k = min(5, len(nearby_indices))
        scores, local_indices = tmp_index.search(embedding, k)
        top = [
            (self.names[nearby_indices[local_indices[0, j]]],
             float(scores[0, j]))
            for j in range(k)
        ]
        log.debug("Landmark top-%d nearby (within %.0f km): %s",
                  k, radius_km,
                  [(n, f"{s:.3f}") for n, s in top])
        if scores[0, 0] >= threshold:
            global_idx = nearby_indices[local_indices[0, 0]]
            return self.names[global_idx], top
        return None, top
