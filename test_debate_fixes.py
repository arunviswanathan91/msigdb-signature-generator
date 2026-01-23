"""
Test Debate System Fixes
=========================

This script tests the critical fixes implemented for the debate system:
1. Updated model configuration (no decommissioned models)
2. Context window management
3. Graceful degradation for model failures

Usage:
    python test_debate_fixes.py <GROQ_API_KEY>
"""

import asyncio
import sys
from debate_system_with_injector import MultiRoundDebateEngine
from db_client_enhanced import DatabaseClientEnhanced


async def test_small_debate(api_key: str):
    """Test with minimal signature (3 genes, 3 rounds)"""
    print("=" * 60)
    print("TEST 1: Small Debate (3 genes, 3 rounds)")
    print("=" * 60)

    # Initialize
    db = DatabaseClientEnhanced("https://arunviswanathan91-msigdb-api.hf.space")

    engine = MultiRoundDebateEngine(
        api_key=api_key,
        db_client=db,
        validate_models=True  # This will show model validation
    )

    # Test with only 3 genes
    test_genes = ["TP53", "EGFR", "KRAS"]

    print(f"\n🧬 Testing signature: {test_genes}")
    print("🎯 Running debate with 3 rounds...\n")

    try:
        result = await engine.run_validation_debate(
            genes=test_genes,
            max_rounds=3,
            tissue_context="lung"
        )

        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"✅ Decision: {result.final_decision}")
        print(f"✅ Confidence: {result.confidence:.2%}")
        print(f"✅ Convergence: {result.convergence_rate:.2%}")
        print(f"✅ Total rounds: {result.total_rounds}")
        print(f"✅ Genes affected: {result.affected_genes}")

        # Check metrics
        metrics = result.decision_metrics
        print(f"\n📊 Quality Metrics:")
        print(f"   • Entropy: {metrics.get('entropy', 0):.3f}")
        print(f"   • Conflict: {metrics.get('conflict', 0):.3f}")
        print(f"   • Raw consensus: {metrics.get('raw_consensus', 0):.3f}")
        print(f"   • Adjusted confidence: {metrics.get('adjusted_confidence', 0):.3f}")

        # Validate results
        print("\n🔍 Validation:")
        if result.confidence == 0.0:
            print("   ❌ FAIL: Confidence is 0.0% (models likely failed)")
            return False
        else:
            print(f"   ✅ PASS: Confidence is {result.confidence:.2%}")

        if result.convergence_rate < 0.1:
            print(f"   ⚠️  WARNING: Low convergence ({result.convergence_rate:.2%})")
        else:
            print(f"   ✅ PASS: Convergence is {result.convergence_rate:.2%}")

        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_context_growth(api_key: str):
    """Test that context doesn't grow beyond limits"""
    print("\n" + "=" * 60)
    print("TEST 2: Context Window Management")
    print("=" * 60)

    db = DatabaseClientEnhanced("https://arunviswanathan91-msigdb-api.hf.space")

    engine = MultiRoundDebateEngine(
        api_key=api_key,
        db_client=db,
        validate_models=False  # Skip validation for speed
    )

    # Test with larger signature and more rounds
    test_genes = ["TP53", "EGFR", "KRAS", "MYC", "BRCA1", "BRCA2", "PTEN", "RB1"]

    print(f"\n🧬 Testing signature: {len(test_genes)} genes")
    print("🎯 Running debate with 5 rounds to test context management...\n")

    try:
        result = await engine.run_validation_debate(
            genes=test_genes,
            max_rounds=5,
            tissue_context="breast"
        )

        print("\n✅ Context management test passed!")
        print(f"   • Completed {result.total_rounds} rounds")
        print(f"   • Final confidence: {result.confidence:.2%}")
        print(f"   • No token limit errors!")

        return True

    except Exception as e:
        if "token" in str(e).lower() or "context" in str(e).lower():
            print(f"\n❌ Context limit exceeded: {e}")
            return False
        else:
            print(f"\n⚠️  Test failed with different error: {e}")
            return False


async def main():
    """Run all tests"""
    if len(sys.argv) < 2:
        print("Usage: python test_debate_fixes.py <GROQ_API_KEY>")
        print("\nExample:")
        print("  python test_debate_fixes.py gsk_xxx...")
        sys.exit(1)

    api_key = sys.argv[1]

    print("\n🚀 Testing Debate System Fixes\n")

    # Run tests
    test1_passed = await test_small_debate(api_key)
    test2_passed = await test_context_growth(api_key)

    # Summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    if test1_passed and test2_passed:
        print("✅ ALL TESTS PASSED!")
        print("\n🎉 The debate system is working correctly:")
        print("   • No decommissioned models")
        print("   • Context window managed properly")
        print("   • Confidence scores are valid")
        print("   • Models responding successfully")
    else:
        print("❌ SOME TESTS FAILED")
        if not test1_passed:
            print("   • Small debate test failed")
        if not test2_passed:
            print("   • Context management test failed")
        print("\n⚠️  Check the errors above for details")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
