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

log = logging.getLogger("landmarks")

DEFAULT_LANDMARKS_PATH = Path.home() / ".local/share/photo-tools/landmarks.json"


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

    def __init__(self, landmarks_path: Path = DEFAULT_LANDMARKS_PATH):
        import faiss

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
        radius_km: float = 50.0,
        threshold: float = 0.55,
    ) -> str | None:
        """Find the best matching landmark for an image embedding.

        Args:
            embedding: L2-normalized CLIP image embedding.
            lat, lon: GPS coordinates of the photo.
            radius_km: Only consider landmarks within this radius of GPS coords.
            threshold: Minimum cosine similarity to return a match.

        Returns:
            Landmark name or None.
        """
        import faiss

        embedding = embedding.reshape(1, -1).astype(np.float32)

        distances = np.array([
            _haversine_km(lat, lon, self.lats[i], self.lons[i])
            for i in range(len(self.landmarks))
        ])
        nearby_indices = np.where(distances <= radius_km)[0]

        if len(nearby_indices) == 0:
            log.debug("No landmarks within %.0f km of (%.4f, %.4f)",
                      radius_km, lat, lon)
            return None

        nearby_embeddings = self._embeddings[nearby_indices]
        dim = nearby_embeddings.shape[1]
        tmp_index = faiss.IndexFlatIP(dim)
        tmp_index.add(nearby_embeddings)

        k = min(5, len(nearby_indices))
        scores, local_indices = tmp_index.search(embedding, k)
        log.debug("Landmark top-%d nearby (within %.0f km): %s",
                  k, radius_km,
                  [(self.names[nearby_indices[local_indices[0, j]]],
                    f"{scores[0, j]:.3f}")
                   for j in range(k)])
        if scores[0, 0] >= threshold:
            global_idx = nearby_indices[local_indices[0, 0]]
            return self.names[global_idx]
        return None
