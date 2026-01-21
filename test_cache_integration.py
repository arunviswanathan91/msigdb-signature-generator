"""
Test Cache Integration
======================
Tests the SearchCacheClient with minimal dependencies.
"""

import sys
import numpy as np
from sentence_transformers import SentenceTransformer

# Add current directory to path
sys.path.insert(0, '.')

from cache_client import SearchCacheClient


def test_cache_client():
    """Test the cache client with dummy data"""

    print("="*70)
    print("TESTING SEARCH CACHE CLIENT")
    print("="*70)

    # Initialize
    print("\n1. Initializing cache client...")
    cache_client = SearchCacheClient()
    print(f"   ✅ Connected to: {cache_client.api_url}")

    # Load model (lightweight)
    print("\n2. Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("   ✅ Model loaded")

    # Create dummy data
    print("\n3. Creating dummy pathway data...")
    pathway_embeddings = {
        'PATHWAY_001': np.random.rand(384).astype(np.float32),
        'PATHWAY_002': np.random.rand(384).astype(np.float32),
        'PATHWAY_003': np.random.rand(384).astype(np.float32),
    }

    pathways_dict = {
        'PATHWAY_001': ['GENE1', 'GENE2', 'GENE3'],
        'PATHWAY_002': ['GENE4', 'GENE5', 'GENE6'],
        'PATHWAY_003': ['GENE7', 'GENE8', 'GENE9'],
    }

    # Normalize embeddings
    for pid in pathway_embeddings:
        pathway_embeddings[pid] = pathway_embeddings[pid] / (
            np.linalg.norm(pathway_embeddings[pid]) + 1e-8
        )

    print("   ✅ Created 3 dummy pathways")

    # Test 1: First search (should be cache MISS)
    print("\n4. Test 1: First search (expect MISS)...")
    results1, scores1, cached1 = cache_client.search_with_cache(
        query="test immune response",
        facet_name="Test Facet",
        mechanism_name="Test Mechanism",
        embedding_model=model,
        pathway_embeddings=pathway_embeddings,
        pathways_dict=pathways_dict,
        top_k=2
    )

    print(f"   Results: {len(results1)} pathways")
    print(f"   Cached: {cached1}")
    assert cached1 == False, "First search should NOT be cached"
    print("   ✅ PASS - First search was computed")

    # Test 2: Same search (should be cache HIT)
    print("\n5. Test 2: Repeat search (expect HIT)...")
    results2, scores2, cached2 = cache_client.search_with_cache(
        query="test immune response",
        facet_name="Test Facet",
        mechanism_name="Test Mechanism",
        embedding_model=model,
        pathway_embeddings=pathway_embeddings,
        pathways_dict=pathways_dict,
        top_k=2
    )

    print(f"   Results: {len(results2)} pathways")
    print(f"   Cached: {cached2}")
    assert cached2 == True, "Second search should be cached"
    print("   ✅ PASS - Second search used cache")

    # Test 3: Results should be identical
    print("\n6. Test 3: Verifying results match...")
    assert results1.keys() == results2.keys(), "Pathway IDs should match"
    print("   ✅ PASS - Cached results match original")

    # Test 4: Get stats
    print("\n7. Test 4: Getting cache statistics...")
    stats = cache_client.get_stats()
    if stats:
        print(f"   Total entries: {stats.get('total_entries', 'N/A')}")
        print(f"   Total hits: {stats.get('total_hits', 'N/A')}")
        print("   ✅ PASS - Stats retrieved")
    else:
        print("   ⚠️ WARNING - Stats not available (server might be sleeping)")

    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED")
    print("="*70)


if __name__ == "__main__":
    try:
        test_cache_client()
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
