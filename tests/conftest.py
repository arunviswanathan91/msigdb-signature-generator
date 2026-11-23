"""Shared fixtures. Every test here is offline and deterministic."""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from signature_engine import KnowledgeIndex  # noqa: E402

TCELL = ["PDCD1", "CTLA4", "LAG3", "HAVCR2", "TIGIT", "TOX", "CD8A", "GZMB",
         "EOMES", "CD160", "CD244", "ENTPD1"]
IFN = ["STAT1", "IRF7", "ISG15", "OAS1", "MX1", "RSAD2", "IFIT1", "IFIT3",
       "IFI6", "OAS2", "IRF9", "STAT2"]


def build_toy_kb(with_housekeeping: bool = False):
    """
    80 gene sets over ~430 genes: 8 T-cell sets, 8 interferon sets, 64 noise.

    Embeddings are 8-d: T-cell sets point along axis 0, interferon sets along
    axis 1, noise sets are random. Returns (pathways, embeddings).
    """
    rng = np.random.default_rng(0)
    background = [f"BG{i}" for i in range(400)]
    pathways, embeddings = {}, {}

    def unit(vector):
        vector = np.asarray(vector, dtype=np.float32)
        return vector / np.linalg.norm(vector)

    for family, members, axis in (("TCELL_EXHAUSTION", TCELL, 0), ("INTERFERON_RESPONSE", IFN, 1)):
        for i in range(8):
            core = list(rng.choice(members, size=9, replace=False))
            extra = list(rng.choice(background, size=3, replace=False))
            genes = core + extra
            if with_housekeeping and family == "TCELL_EXHAUSTION":
                genes = genes + ["GAPDH"]
            pathways[f"{family}_{i}"] = genes
            direction = np.zeros(8)
            direction[axis] = 1.0
            embeddings[f"{family}_{i}"] = unit(direction + 0.05 * rng.standard_normal(8))

    for i in range(64):
        pathways[f"NOISE_{i}"] = list(rng.choice(background, size=20, replace=False))
        embeddings[f"NOISE_{i}"] = unit(np.r_[0.0, 0.0, rng.standard_normal(6)])

    return pathways, embeddings


@pytest.fixture(scope="session")
def toy_index():
    pathways, embeddings = build_toy_kb()
    return KnowledgeIndex(pathways, embeddings)


@pytest.fixture(scope="session")
def toy_index_housekeeping():
    pathways, embeddings = build_toy_kb(with_housekeeping=True)
    return KnowledgeIndex(pathways, embeddings)


def axis_encoder(text: str) -> np.ndarray:
    """Fake sentence encoder: 'interferon' -> axis 1, everything else -> axis 0."""
    vector = np.zeros(8, dtype=np.float32)
    vector[1 if "interferon" in text.lower() else 0] = 1.0
    return vector
