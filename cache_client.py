"""
Search Cache Client
===================
Client for interacting with the search cache API on HuggingFace.

This client wraps semantic search operations with automatic caching.
"""

import requests
import numpy as np
from typing import Dict, List, Optional, Tuple
from sentence_transformers import SentenceTransformer


class SearchCacheClient:
    """
    Client for semantic search cache.

    Automatically caches and retrieves search results to speed up
    repeated queries.
    """

    def __init__(self, api_url: str = "https://arunviswanathan91-msigdb-api.hf.space"):
        """
        Initialize cache client.

        Args:
            api_url: Base URL of the cache API server
        """
        self.api_url = api_url.rstrip('/')
        self.kb_version = "2025.1"  # Update when KB changes
        self.session = requests.Session()

        # Configure session
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })

    def search_with_cache(
        self,
        query: str,
        facet_name: str,
        mechanism_name: str,
        embedding_model: SentenceTransformer,
        pathway_embeddings: Dict[str, np.ndarray],
        pathways_dict: Dict[str, List[str]],
        top_k: int = 50
    ) -> Tuple[Dict[str, List[str]], Dict[str, float], bool]:
        """
        Perform semantic search with automatic caching.

        Workflow:
        1. Check cache for existing results
        2. If found, return cached results (FAST)
        3. If not found, compute search (SLOW)
        4. Store results in cache for future use

        Args:
            query: Search query text
            facet_name: Name of the biological facet
            mechanism_name: Mechanism/pathway name
            embedding_model: Sentence transformer model
            pathway_embeddings: Pre-computed pathway embeddings
            pathways_dict: Pathway ID -> gene list mapping
            top_k: Number of top results to return

        Returns:
            Tuple of:
            - selected_pathways: Dict[pathway_id, gene_list]
            - similarity_scores: Dict[pathway_id, score]
            - was_cached: Boolean indicating if result was from cache
        """
        # Step 1: Check cache
        cached_result = self._query_cache(facet_name, mechanism_name)

        if cached_result and cached_result.get('cached'):
            # Cache HIT - reconstruct results
            pathway_ids = cached_result['pathway_ids']
            similarity_scores = cached_result['similarity_scores']

            # Reconstruct pathways dict
            selected_pathways = {
                pid: pathways_dict[pid]
                for pid in pathway_ids
                if pid in pathways_dict
            }

            return selected_pathways, similarity_scores, True

        # Step 2: Cache MISS - perform search
        # Encode query
        query_emb = embedding_model.encode(query, convert_to_numpy=True)
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)

        # Compute similarities
        similarities = []
        for pid, pathway_emb in pathway_embeddings.items():
            similarity = float(np.dot(query_emb, pathway_emb))
            similarities.append((pid, similarity))

        # Sort and select top K
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_pathways = similarities[:top_k]

        # Build results
        selected_pathways = {
            pid: pathways_dict[pid]
            for pid, _ in top_pathways
            if pid in pathways_dict
        }
        similarity_dict = {pid: score for pid, score in top_pathways}

        # Step 3: Store in cache (asynchronously, don't wait)
        self._store_in_cache(
            facet_name=facet_name,
            mechanism_name=mechanism_name,
            query_embedding=query_emb,
            pathway_ids=list(similarity_dict.keys()),
            similarity_scores=similarity_dict
        )

        return selected_pathways, similarity_dict, False

    def _query_cache(self, facet: str, mechanism: str) -> Optional[Dict]:
        """
        Query cache via API.

        Args:
            facet: Facet name
            mechanism: Mechanism name

        Returns:
            Cache result dict or None if not found/error
        """
        try:
            response = self.session.post(
                f"{self.api_url}/api/cache/search",
                json={
                    "facet": facet,
                    "mechanism": mechanism,
                    "kb_version": self.kb_version
                },
                timeout=5
            )

            if response.status_code == 200:
                return response.json()
            else:
                return None

        except Exception as e:
            # Silently fail - cache is optional
            return None

    def _store_in_cache(
        self,
        facet_name: str,
        mechanism_name: str,
        query_embedding: np.ndarray,
        pathway_ids: List[str],
        similarity_scores: Dict[str, float]
    ) -> bool:
        """
        Store search result in cache.

        This is non-blocking and failures are ignored (cache is optional).

        Args:
            facet_name: Facet name
            mechanism_name: Mechanism name
            query_embedding: Query embedding vector
            pathway_ids: List of pathway IDs
            similarity_scores: Dict of pathway_id -> score

        Returns:
            True if stored successfully, False otherwise
        """
        try:
            response = self.session.post(
                f"{self.api_url}/api/cache/store",
                json={
                    "facet": facet_name,
                    "mechanism": mechanism_name,
                    "kb_version": self.kb_version,
                    "query_embedding": query_embedding.tolist(),
                    "pathway_ids": pathway_ids,
                    "similarity_scores": similarity_scores
                },
                timeout=10
            )

            return response.status_code == 200

        except Exception as e:
            # Silently fail - cache storage is optional
            return False

    def get_stats(self) -> Optional[Dict]:
        """
        Get cache statistics from API.

        Returns:
            Stats dict with total_entries, total_hits, kb_versions, etc.
            Or None if request fails.
        """
        try:
            response = self.session.get(
                f"{self.api_url}/api/cache/stats",
                timeout=5
            )

            if response.status_code == 200:
                return response.json()
            return None

        except Exception:
            return None
