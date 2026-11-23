#!/usr/bin/env python
"""
Build or verify data/pathway_embeddings.pkl
===========================================

The semantic channel compares a mechanism description with one embedding per
gene set. This script regenerates those embeddings from the knowledge base, so
the file in ``data/`` is reproducible rather than an opaque artefact.

Text embedded for each gene set (verified to reproduce the committed file to
cosine 1.0000):

    <name with "_" replaced by " "> <first 10 genes, space-separated>

Model: sentence-transformers/all-MiniLM-L6-v2, L2-normalised, 384 dimensions.

    python scripts/build_embeddings.py --verify            # check the committed file
    python scripts/build_embeddings.py --verify --sample 0 # check every gene set
    python scripts/build_embeddings.py --out data/pathway_embeddings.pkl
"""

import argparse
import gzip
import json
import pickle
import sys
from datetime import datetime

import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"
GENES_IN_TEXT = 10


def embedding_text(name: str, genes) -> str:
    return f"{name.replace('_', ' ')} {' '.join(genes[:GENES_IN_TEXT])}"


def encode(texts, batch_size: int = 128) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)
    return model.encode(texts, normalize_embeddings=True, batch_size=batch_size,
                        show_progress_bar=len(texts) > 2000).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--kb", default="data/knowledge_base.json.gz")
    parser.add_argument("--embeddings", default="data/pathway_embeddings.pkl",
                        help="existing file to verify against")
    parser.add_argument("--verify", action="store_true",
                        help="compare freshly computed embeddings with --embeddings")
    parser.add_argument("--sample", type=int, default=1000,
                        help="gene sets to verify (0 = all)")
    parser.add_argument("--out", default=None, help="write a new embeddings file here")
    args = parser.parse_args()
    if not args.verify and not args.out:
        parser.error("choose --verify and/or --out")

    with gzip.open(args.kb, "rt", encoding="utf-8") as handle:
        kb = json.load(handle)
    pathways = kb["pathways"]
    names = list(pathways)

    if args.verify:
        with open(args.embeddings, "rb") as handle:
            package = pickle.load(handle)
        stored = package["embeddings"]
        missing = [n for n in names if n not in stored]
        extra = [n for n in stored if n not in pathways]
        print(f"gene sets in KB: {len(names):,} | embeddings: {len(stored):,} | "
              f"missing: {len(missing)} | not in KB: {len(extra)}")
        rng = np.random.default_rng(0)
        chosen = names if args.sample <= 0 or args.sample >= len(names) else \
            [names[i] for i in rng.choice(len(names), args.sample, replace=False)]
        chosen = [n for n in chosen if n in stored]
        fresh = encode([embedding_text(n, pathways[n]) for n in chosen])
        cosine = np.sum(fresh * np.stack([stored[n] for n in chosen]), axis=1)
        print(f"verified {len(chosen):,} gene sets: mean cosine {cosine.mean():.6f}, "
              f"min {cosine.min():.6f}")
        if missing or extra or cosine.min() < 0.999:
            print("MISMATCH: the embeddings file does not correspond to this knowledge base")
            return 1
        print("OK: embeddings match the knowledge base and the documented recipe")

    if args.out:
        vectors = encode([embedding_text(n, pathways[n]) for n in names])
        package = {
            "embeddings": {n: vectors[i] for i, n in enumerate(names)},
            "model_name": MODEL_NAME,
            "embedding_dim": int(vectors.shape[1]),
            "normalized": True,
            "metadata": {
                "created": datetime.now().isoformat(),
                "kb_version": kb.get("metadata", {}).get("version"),
                "kb_pathways": len(names),
                "total_pathways": len(names),
                "text_recipe": "name with '_' as spaces + first 10 genes, space-joined",
            },
        }
        with open(args.out, "wb") as handle:
            pickle.dump(package, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"wrote {len(names):,} embeddings to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
